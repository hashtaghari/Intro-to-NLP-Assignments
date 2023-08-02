- to run the code just run the following command:
- python3 svd.py - for SVD implementation
- python3 cbow.py - for CBOW implementation
- 2d plots for comparison are in the report
- the python files themselves load for the best model once you download it from the google drive given in the report
- the pre-trained embedding for comparison is also in the google drive

# Explanation of the code - SVD.py
The implementation is a Python script that uses the Gensim and NLTK libraries to preprocess and tokenize text data, and the scikit-learn and NumPy libraries to perform matrix operations and dimensionality reduction. The script reads a subset of movie reviews in JSON format from a file, preprocesses and tokenizes the text data, and creates a vocabulary of words that appear at least three times in the corpus. It then constructs a co-occurrence matrix of word pairs within a window of five words, and performs Singular Value Decomposition (SVD) on the matrix to obtain a matrix of word embeddings.
The script loads the resulting matrix of word embeddings, and uses it to find the top 10 closest words to a set of target words using cosine similarity. It then visualizes the resulting embeddings of the target words and their closest neighbors in two-dimensional space using t-SNE.
Overall, the implementation is a basic example of how to construct and visualize word embeddings using a co-occurrence matrix and SVD.

# Explanation of the code - CBOW.py
The code first loads a dataset containing movie reviews and preprocesses it by removing stop words and infrequent words. It then tokenizes the text using Keras's Tokenizer and maps each token to a unique integer index.
Next, it generates context-target pairs for each word in the text using a specified window size, which is then padded to a fixed length. The input and output pairs are saved in two separate files.
The CBOW model is then defined using PyTorch's nn module, which consists of an embedding layer, two linear layers, and activation functions. The input is a context window of words, which are embedded into a lower-dimensional space. The resulting embeddings are summed, passed through the two linear layers, and then passed through a softmax activation function to output a probability distribution over all words in the vocabulary.
The model is then trained using the negative log likelihood loss function and stochastic gradient descent (SGD) optimizer. The training loop iterates over the input and output pairs, updates the model parameters, and computes the loss. The progress of the training is visualized using the tqdm library.