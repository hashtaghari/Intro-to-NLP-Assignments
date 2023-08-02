import re

def tokenize_line(line):

    if line != '':
        line = line.strip()
        line = line.lower()

    # remove punctuation except < or >
    line = re.sub(r"[^\w\s]", "", line)

    # remove Non-ASCII character
    line = re.sub(r'[^\x00-\x7F]+', '', line)

    # add <start> 
    line = "<s> <s> <s> " +  line + " </s> </s> </s>"
        
    # replace URLs with <URL>
    line = re.sub(r"http\S+", "<URL>", line)

    # replace URLs that start with www to <URL>
    line = re.sub(r"www\S+", "<URL>", line)

    # replace HASHTAGS with <HASHTAGS>
    line = re.sub(r"#\S+", "<HASHTAG>", line)

    # replace MENTIONS with <MENTION>
    line = re.sub(r"@\S+", "<MENTION>", line)

    # handle emails
    line = re.sub(r"\S+@\S+", "<EMAIL>", line)

    # handle time 
    line = re.sub(r"\d+:\d+", "<TIME>", line)

    # handle dates
    line = re.sub(r"\d+/\d+/\d+", "<DATE>", line)

    # replace numbers with <NUM>
    line = re.sub(r"\d+", "<NUM>", line)

    # replace words containing numbers to <NUM>
    line = re.sub(r"\w*\d\w*", "<NUM>", line)


    # substitute word contractions with original words
    line = re.sub(r"i'm", "i am", line)
    line = re.sub(r"he's", "he is", line)
    line = re.sub(r"she's", "she is", line)
    line = re.sub(r"that's", "that is", line)
    line = re.sub(r"what's", "what is", line)
    line = re.sub(r"where's", "where is", line)
    line = re.sub(r"\'ll", " will", line)
    line = re.sub(r"\'ve", " have", line)
    line = re.sub(r"\'re", " are", line)
    line = re.sub(r"\'d", " would", line)
    line = re.sub(r"won't", "will not", line)
    line = re.sub(r"can't", "cannot", line)
    line = re.sub(r"n't", " not", line)

    

    # remove extra spaces
    line = re.sub(r"\s+", " ", line)

    

    # tokenize
    tokens = line.split()

    return tokens


# helper functions
def converttoUNK(tokenized_dataset):
    # convert words with freq <= 3 into <UNK>
    word_freq = {}
    for line in tokenized_dataset:
        for token in line:
            if token not in word_freq:
                word_freq[token] = 1
            else:
                word_freq[token] += 1

    for i in range(len(tokenized_dataset)):
        for j in range(len(tokenized_dataset[i])):
            if word_freq[tokenized_dataset[i][j]] <= 3:
                tokenized_dataset[i][j] = '<UNK>'
    return tokenized_dataset

    