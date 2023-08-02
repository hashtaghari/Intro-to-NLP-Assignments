
## HOW TO RUN THE CODE?
- run the pos_tagger.py file
- the pre-trained model is saved in the same directory as the above file named as 'best_model.pth'


## About Code
This code is an implementation of a neural part-of-speech (POS) tagger using PyTorch. A POS tagger is a tool that assigns a grammatical tag to each word in a sentence, indicating its part of speech, such as noun, verb, adjective, or adverb. This can be useful in natural language processing tasks such as parsing, machine translation, and sentiment analysis.

The code is divided into three main sections: vocabulary creation, data preparation, and model implementation.

## Vocabulary Creation
In the vocabulary creation section, the Vocab class is defined. This class is responsible for creating the vocabulary for both the input tokens and the output POS tags. The __init__ method takes an iterable object, which is expected to contain the tokens or POS tags. The vocabulary contains two special tokens, <pad> and <unk>, which are used for padding and out-of-vocabulary tokens, respectively. The create_idx_2_word method creates a list of tokens sorted by their frequency, and the create_word_2_idx method creates a dictionary that maps each token to its index in the list. The word2id and id2word methods are used to convert between tokens and their corresponding indices.

## Data Preparation
In the data preparation section, the CoNLLDataset class is defined. This class reads in a CoNLL-formatted file (a common format for annotated corpora in NLP) and extracts the tokens and POS tags. It also creates vocabularies for the tokens and POS tags using the Vocab class defined earlier. The __getitem__ method takes an index and returns the input and target sequences for that index. The input sequence is a list of token indices, and the target sequence is a list of POS tag indices. The collate_annotations function is used to pad the input and target sequences to a common length, sort the sequences by length for efficiency, and convert the sequences to PyTorch tensors. The DataLoader class is used to create batches of data for training and validation.

Finally, the code defines hyperparameters such as the batch size, number of epochs, and optimizer. It trains the model on the training dataset and evaluates its performance on the validation dataset. It uses several metrics such as accuracy, recall, precision, and F1 score to evaluate the model's performance.
