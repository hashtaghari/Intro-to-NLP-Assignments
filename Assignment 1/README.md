## Tokenizer
- handles all the basic required cases like URLs, HashTags, Mentions, repeated punctuations by cleaning the corpus
- also handles extra cases like removes Non-ASCII characters, adds <s> and </s> [somewhat like padding] to every sentence, handles numbers, emails, dates, time and word contractions, as well as removes trailing and leading spaces
- finally, it tokenizes the cleaned data and returns the tokens

## Smoothing
- In general, first we perform line by line tokenization [lines created by \n as delimiter].
- then the tokens are stored in a tokenized dataset
- we randomly take out 1000 sentences into test set and remaining into train set
- we then handle the case of words occuring very rarely and convert them into <UNK> tokens (here we took the threshold frequency to be 3)
- to calculate perplexity of the train and test data, perplexity function is used which uses the standard formula for calculating perplexities [commented out] [perplexity data stored in text files]
- to calculate perplexity we use smoothing techniques
- an input prompt also pops out as per the requirements of the assignment
- the probability is output which is calculated using 4 gram LM + smoothing techniques
- the smoothing techniques used are:
    - Kneser-Ney smoothing
    - Witten-Bell smoothing
- the perplexity scores are calculated using both these techniques for both the given corpora.

