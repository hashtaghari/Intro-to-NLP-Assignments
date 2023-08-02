
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from nltk.tokenize import word_tokenize
import pandas as pd
import json
import re
import string
from nltk.corpus import stopwords

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.linalg import norm
from scipy import sparse
import numpy as np

path = './data/reviews_Movies_and_TV.json'

# load first 100000 reviews that too only reviewText
data = pd.read_json(path, lines=True, nrows=100000)[['reviewText']]

# store the data in a file as list
data.to_csv('./data/reviews.csv', index=False, header=False)

# print the first 5 reviews
data.head()

# tokenize the data


def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    text = text.split()
    return text


# tokenize
data['reviewText'] = data['reviewText'].apply(tokenize)

# remove stop words
stop_words = stopwords.words('english')
data['reviewText'] = data['reviewText'].apply(
    lambda x: [word for word in x if word not in stop_words])


# create vocab
# remove words with frequency less than 4
vocab = {}
for row in data['reviewText']:
    for word in row:
        if word in vocab:
            vocab[word] += 1
        else:
            vocab[word] = 1

# remove words with frequency less than 4
vocab = {k: v for k, v in vocab.items() if v > 2}

# create a list of words
vocab = list(vocab.keys())

# create a dictionary of words and their index
vocab = {word: idx for idx, word in enumerate(vocab)}

# print the first 5 words
list(vocab.items())[:5]

# print('Vocab size: ', len(vocab))
# save the vocab
with open('./data/vocab.json', 'w') as f:
    json.dump(vocab, f)

# create a dictionary to store the word indices
word_indices = {}

# create a dictionary to store the index words
index_words = {}

# iterate over the vocab and create the dictionaries
for index, word in enumerate(vocab):
    word_indices.update({word: index})
    index_words.update({index: word})

# save the dictionaries
with open('./data/word_indices.json', 'w') as f:
    json.dump(word_indices, f)

with open('./data/index_words.json', 'w') as f:
    json.dump(index_words, f)

# # create a co-occurence matrix
# # we will use a sparse matrix to save memory

# create a sparse matrix of zeros
# co_occurence_matrix = sparse.lil_matrix((len(vocab), len(vocab)))

# # iterate over the reviews
# for review in data['reviewText']:
#     # create a list of indices of the words in the review
#     indices = [word_indices[word] for word in review if word in word_indices]
#     # iterate over the indices and update the co-occurence matrix in the window = 5
#     for i, index in enumerate(indices):
#         # get the words in the window
#         window = indices[max(0, i-5):i] + indices[i+1:min(len(indices), i+6)]
#         # iterate over the words in the window
#         for other_index in window:
#             # increment the co-occurence count
#             co_occurence_matrix[index, other_index] += 1

# # convert the co-occurence matrix to csr format
# co_occurence_matrix_csr = co_occurence_matrix.tocsr()

# # save the co-occurence matrix
# sparse.save_npz('./data/co_occurence_matrix.npz', co_occurence_matrix_csr)

# # Perform SVD

# # import the co-occurence matrix
# co_occurence_matrix = sparse.load_npz('./data/co_occurence_matrix.npz')

# # import the norm function from numpy.linalg


# # perform SVD
# k = 300
# u, s, v = sparse.linalg.svds(co_occurence_matrix, k=k)

# # save the u matrix
# np.save('./data/u.npy', u)

# load U matrix
u = np.load('./data/u.npy')

# load the word indices
with open('./data/word_indices.json', 'r') as f:
    word_indices = json.load(f)

# load the index words
with open('./data/index_words.json', 'r') as f:
    index_words = json.load(f)

words = ["sometimes", "know", "media", "china", "pleased"]

# get the top 10 closest words and plot them using t-SNE as 2d plots
for word in words:
    if word in word_indices:
        print(word)
        # get the index of the word
        word_index = word_indices[word]
        # get the word vector
        word_vector = u[word_index]
        # calculate the cosine similarity
        cosine_similarities = u.dot(word_vector) / \
            (norm(u, axis=1) * norm(word_vector))

        # get the top 10 closest words
        top_10 = np.argsort(cosine_similarities)[-10:]
        print(top_10)
        # get the word vectors of the top 10 closest words
        word_vectors = u[top_10]
        # reduce the dimensionality of the word vectors to 2d
        tsne = TSNE(n_components=2, random_state=0, perplexity=5)
        np.set_printoptions(suppress=True)
        Y = tsne.fit_transform(word_vectors)
        # plot the 2d word vectors
        x_coords = Y[:, 0]
        y_coords = Y[:, 1]
        plt.scatter(x_coords, y_coords)
        for label, x, y in zip(top_10, x_coords, y_coords):
            plt.annotate(index_words[str(label)], xy=(x, y), xytext=(0, 0),
                         textcoords='offset points')
        plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
        plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
        plt.show()


# using pre-trained word vectors get top 10 closest words to 'titanic'
# get top 10 closest words to 'titanic' using U
# compare them using plot

# get the word vector
word_index = word_indices['titanic']
word_vector = u[word_index]
# calculate the cosine similarity
cosine_similarities = u.dot(word_vector) / \
    (norm(u, axis=1) * norm(word_vector))

# get the top 10 closest words
top_10 = np.argsort(cosine_similarities)[-10:]
# get the word vectors of the top 10 closest words
word_vectors = u[top_10]
# reduce the dimensionality of the word vectors to 2d
tsne = TSNE(n_components=2, random_state=0, perplexity=5)
np.set_printoptions(suppress=True)
Y = tsne.fit_transform(word_vectors)
# plot the 2d word vectors
x_coords = Y[:, 0]
y_coords = Y[:, 1]
plt.scatter(x_coords, y_coords)
for label, x, y in zip(top_10, x_coords, y_coords):
    plt.annotate(index_words[str(label)], xy=(x, y), xytext=(0, 0),
                 textcoords='offset points')
plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
plt.show()


# convert GloVe file to word2vec format
glove_file = './glove.6B.100d.txt'
word2vec_file = './glove.6B.100d.word2vec'
glove2word2vec(glove_file, word2vec_file)

# load the pre-trained GloVe vectors in word2vec format
word_vectors = KeyedVectors.load_word2vec_format(word2vec_file, binary=False)

# get the top 10 closest words
top_10 = word_vectors.most_similar('titanic', topn=10)
# get the word vectors of the top 10 closest words
word_vectors = np.array([word_vectors[word] for word, _ in top_10])
tsne = TSNE(n_components=2, random_state=0, perplexity=5)
np.set_printoptions(suppress=True)
Y = tsne.fit_transform(word_vectors)
# plot the 2d word vectors
x_coords = Y[:, 0]
y_coords = Y[:, 1]
plt.scatter(x_coords, y_coords)
for label, x, y in zip(top_10, x_coords, y_coords):
    plt.annotate(label[0], xy=(x, y), xytext=(0, 0),
                 textcoords='offset points')
plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
plt.show()
