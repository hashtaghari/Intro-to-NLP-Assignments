import torch
import numpy as np
import os
from collections import Counter
import re
from torch.utils.data import Dataset
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import string
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

"""## Vocabulary"""

class Vocab(object):
    def __init__(self, iter):
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self._id2word = self.create_idx_2_word(iter)
        self._word2id = self.create_word_2_idx(iterable=self._id2word)

    def create_idx_2_word(self, iter):
        # Add special tokens.
        id2word = [self.pad_token]
        id2word.append(self.unk_token)
        # Create dictionary to store token counts
        token_counts = {}
        # Update dictionary with token counts
        for x in iter:
            for token in x:
                if token in token_counts:
                    token_counts[token] += 1
                else:
                    token_counts[token] = 1
        # Sort tokens by frequency in descending order
        sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
        # Add words to vocabulary
        for token, count in sorted_tokens:
            id2word.append(token)
        return id2word

    def create_word_2_idx(self, iterable):
        # Create word to id mapping
        word2id = {x: i for i, x in enumerate(iterable)}
        return word2id

    def __len__(self):
        return len(self._id2word)

    def word2id(self, word):
        if word in self._word2id:
            return self._word2id[word]
        else:
            return self._word2id[self.unk_token]

    def id2word(self, id):
        return self._id2word[id]

"""## Data Prep"""

class CoNLLDataset(Dataset):
    def __init__(self, fname):
        self.data = open(fname, 'r').read()
        self.tokens, self.pos_tags = self.extract_fields(self.data)
        self.token_vocab = Vocab([x for x in self.tokens])
        self.pos_vocab = Vocab([x for x in self.pos_tags])

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        tokens = self.tokens[idx]
        pos_tags = self.pos_tags[idx]
        input = [self.token_vocab.word2id(x) for x in tokens]
        target = [self.pos_vocab.word2id(x) for x in pos_tags]
        return input, target
    
    def extract_fields(self, raw_text):
        # Split into chunks on blank lines.
        conll_Lines = re.split(r'^\n', raw_text, flags=re.MULTILINE)
        # Process each chunk into an annotation.
        tokens = []
        pos_tags = []
        for conll_Line in conll_Lines:
            token = []
            pos_tag = []
            lines = conll_Line.split('\n')
            for line in lines:
                # If line is empty ignore it.
                if len(line) == 0 or line[0] =='#':
                    continue
                fields = line.split('\t')
                token.append(fields[1])
                pos_tag.append(fields[3])
            if (len(token) > 0) and (len(pos_tag) > 0):
                tokens.append(token)
                pos_tags.append(pos_tag)
        return tokens, pos_tags

def collate_annotations(batch):
    # Get inputs, targets, and lengths.
    inputs, targets = zip(*batch)
    lengths = [len(x) for x in inputs]
    # Sort by length.
    sorted_indices = sorted(range(len(lengths)), key=lengths.__getitem__, reverse=True)
    inputs = [inputs[i] for i in sorted_indices]
    targets = [targets[i] for i in sorted_indices]
    lengths = [lengths[i] for i in sorted_indices]
    # Pad.
    max_length = max(lengths)
    temp = [input + [0]*(max_length - len(input)) for input in inputs]
    inputs = temp
    temp = [target + [0]*(max_length - len(target)) for target in targets]
    targets = temp
    # Transpose.
    inputs = list(map(list, zip(*inputs)))
    targets = list(map(list, zip(*targets)))
   
    inputs, targets, lengths = torch.LongTensor(inputs).to(device), torch.LongTensor(targets).to(device), torch.LongTensor(lengths).to(device)

    return inputs, targets, lengths

# Load datasets.
train_dataset = CoNLLDataset('en_atis-ud-train.conllu')
dev_dataset = CoNLLDataset('en_atis-ud-dev.conllu')

dev_dataset.token_vocab = train_dataset.token_vocab
dev_dataset.pos_vocab = train_dataset.pos_vocab

input_vocab_size = len(train_dataset.token_vocab)
output_vocab_size = len(train_dataset.pos_vocab)

batch_size = 100
epochs = 10

data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                         collate_fn=collate_annotations)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=collate_annotations)

"""## Model Implementation"""

class NeuralPOSTagger(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, embedding_dim=200, hidden_size=100,
                 bidirectional=True):
        super(NeuralPOSTagger, self).__init__()
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.word_embeddings = nn.Embedding(input_vocab_size, embedding_dim,
                                            padding_idx=0)
        self.rnn = nn.GRU(embedding_dim, hidden_size,
                          bidirectional=bidirectional)
        self.fc = nn.Linear(2*hidden_size if bidirectional else hidden_size, output_vocab_size)
        self.activation = nn.LogSoftmax(dim=2)

    def init_hidden(self, batch_size):
            hidden = torch.zeros(2 if self.bidirectional else 1, batch_size, self.hidden_size)
            return hidden.to(device)

    def forward(self, x, lengths=None, hidden=None):
        seq_len, batch_size = x.size()
        if hidden is None:
           hidden = self.init_hidden(batch_size)

        net = self.word_embeddings(x)
        # Pack before feeding into the RNN.
        if lengths is not None:
            lengths = lengths.data.view(-1).tolist()
            net = pack_padded_sequence(net, lengths)
        net, hidden = self.rnn(net, hidden)
        # Unpack after
        if lengths is not None:
            net, _ = pad_packed_sequence(net)
        net = self.fc(net)
        net = self.activation(net)

        return net, hidden

"""## Training"""

def train():
    # Initialize the model.
    model = NeuralPOSTagger(input_vocab_size, output_vocab_size)
    model.to(device)

    # Initialize loss function and optimizer.
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    # if best model exists , load it
    if os.path.isfile('best_model.pth'):
        model = torch.load('best_model.pth')

    train_losses = []
    dev_losses = []
    i = 0
    min_dev_loss = 1000000
    for epoch in range(epochs):
        for inputs, targets, lengths in data_loader:
            model.zero_grad()
            outputs, _ = model(inputs, lengths=lengths)

            outputs = outputs.view(-1, output_vocab_size)
            targets = targets.view(-1)

            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.data)
            if (i % 50) == 0:
                for inputs, targets, lengths in dev_loader:
                    outputs, _ = model(inputs, lengths=lengths)
                    outputs = outputs.view(-1, output_vocab_size)
                    targets = targets.view(-1)
                    loss = loss_function(outputs, targets)
                    dev_losses.append(loss.data)
                
                train_loss_sum = 0
                train_batches = 0
                for train_loss in train_losses:
                    train_loss_sum += train_loss.cpu().item()
                    train_batches += 1
                avg_train_loss = train_loss_sum / train_batches

                dev_loss_sum = 0
                dev_batches = 0
                for dev_loss in dev_losses:
                    dev_loss_sum += dev_loss.cpu().item()
                    dev_batches += 1
                avg_dev_loss = dev_loss_sum / dev_batches

                print("Overall iteration: ", i)
                print("Training Loss: ", avg_train_loss)
                print("Dev Set Loss: ", avg_dev_loss)
                # save best model according to min dev loss
                if avg_dev_loss < min_dev_loss:
                    torch.save(model, 'best_model.pth')
                    min_dev_loss = avg_dev_loss
                train_losses = []
                dev_losses = []
                torch.save(model, 'pos_tagger.pth')
            i += 1
        
        # save model after each epoch in saved_models folder and name it accordingly
        torch.save(model, 'saved_models/pos_tagger_epoch_' + str(epoch) + '.pth')

"""## Training Scores"""

def training_scores():
    # Get accuracy, recall, score, f1 score on dev set
    model = torch.load('best_model.pth')
    model.to(device)

    # Compute accuracy, recall, precision, f1 score on dev set
    

    # Initialize lists to store predictions and targets.
    y_pred = []
    y_true = []

    # Iterate over the dev set.
    for inputs, targets, lengths in dev_loader:
        # Compute predictions.
        outputs, _ = model(inputs, lengths=lengths)
        outputs = outputs.view(-1, output_vocab_size)
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.cpu().numpy()
        targets = targets.view(-1).cpu().numpy()

        # Store predictions and targets.
        y_pred.append(predicted)
        y_true.append(targets)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    # Compute accuracy
    acc = np.mean(y_true[y_true != 0] == y_pred[y_true != 0])
    print('Accuracy: ', acc)

    # Compute recall
    recall = recall_score(y_true[y_true != 0], y_pred[y_true != 0], average='macro')
    print('Recall: ', recall)

    # Compute precision
    precision = precision_score(y_true[y_true != 0], y_pred[y_true != 0], average='macro')
    print('Precision: ', precision)

    # Compute f1 score
    f1 = f1_score(y_true[y_true != 0], y_pred[y_true != 0], average='macro')
    print('F1 Score: ', f1)

    # get classification report
    from sklearn.metrics import classification_report
    print(classification_report(y_true[y_true != 0], y_pred[y_true != 0], target_names=train_dataset.pos_vocab._id2word))

"""## Testing

"""

def Testing():
    # Get accuracy, recall, score, f1 score
    def get_metrics(y_true, y_pred):
        # Compute accuracy
        accuracy = np.mean(y_true[y_true != 0] == y_pred[y_true != 0])
        # Compute recall
        recall = recall_score(y_true[y_true != 0],
                            y_pred[y_true != 0], average='macro')
        # Compute precision
        precision = precision_score(
            y_true[y_true != 0], y_pred[y_true != 0], average='macro')
        # Compute f1 score
        f1 = f1_score(y_true[y_true != 0], y_pred[y_true != 0], average='macro')
        return accuracy, recall, precision, f1

    # Load the best model
    model = torch.load('best_model.pth')
    model.to(device)

    # Load the test set.
    test_dataset = CoNLLDataset('en_atis-ud-test.conllu')
    test_dataset.token_vocab = train_dataset.token_vocab
    test_dataset.pos_vocab = train_dataset.pos_vocab

    # Initialize the data loader.
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                collate_fn=collate_annotations)

    # Initialize the lists to store the predictions and true labels.
    y_true = []
    y_pred = []

    # Get the predictions.
    for inputs, targets, lengths in test_loader:
        outputs, _ = model(inputs, lengths=lengths)
        _, predicted = torch.max(outputs, 2)
        predicted = predicted.view(-1)
        targets = targets.view(-1)
        y_true.append(targets.data.cpu().numpy())
        y_pred.append(predicted.data.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    # Get the metrics.
    accuracy, recall, precision, f1 = get_metrics(y_true, y_pred)
    print("Accuracy: ", accuracy)
    print("Recall: ", recall)
    print("Precision: ", precision)
    print("F1 score: ", f1)

# train()
# training_scores()
# Testing()

# sentence = input("Enter a sentence: ")
sentence = "what are the coach flights between dallas and baltimore leaving august tenth and returning august twelve"
# convert to lower case
sentence = sentence.lower()

# remove punctuations
sentence = sentence.translate(str.maketrans('', '', string.punctuation))

# tokenize sentence using Vocab, Dataset
sentence = sentence.split()



# using input and best model, give POS tags as output

# load best model
model = torch.load('best_model.pth')
model.to(device)

# prepare sentence as an input to model

# convert sentence to ids
sentence_ids = []
for word in sentence:
    sentence_ids.append(train_dataset.token_vocab.word2id(word))

# convert to tensor
sentence_ids = torch.tensor(sentence_ids).to(device)

# convert to batch
sentence_ids = sentence_ids.unsqueeze(0)
print(sentence_ids)

# get lengths
lengths = torch.tensor([len(sentence_ids)]).to(device)

# get predictions
outputs, _ = model(sentence_ids)
_, predicted = torch.max(outputs, 2)
predicted = predicted.view(-1)


# convert to list
predicted = predicted.tolist()

# convert to words
predicted_words = []
for tag in predicted:
    predicted_words.append(train_dataset.pos_vocab.id2word(tag))

# print output words vs POS
for i in range(len(sentence)):
    print(sentence[i], predicted_words[i])
