# library imports
import sys
import random
import math
# tokenizer
from tokenizer_helper import tokenize_line, converttoUNK


def get_unigrams(tokens):
	unigrams = {}
	for sentence in tokens:
		for word in sentence:
			if word not in unigrams.keys():
				unigrams[word] = 1
			else:
				unigrams[word] += 1
	return unigrams


def get_bigrams(tokens):
	bigrams = {}
	for sentence in tokens:
		for i in range(len(sentence)-1):
			if sentence[i] not in bigrams.keys():
				bigrams[sentence[i]] = {}
			if sentence[i+1] not in bigrams[sentence[i]].keys():
				bigrams[sentence[i]][sentence[i+1]] = 1
			else:
				bigrams[sentence[i]][sentence[i+1]] += 1
	return bigrams


def get_trigrams(tokens):
	trigrams = {}
	for sentence in tokens:
		for i in range(len(sentence)-2):
			if sentence[i] not in trigrams.keys():
				trigrams[sentence[i]] = {}
			if sentence[i+1] not in trigrams[sentence[i]].keys():
				trigrams[sentence[i]][sentence[i+1]] = {}
			if sentence[i+2] not in trigrams[sentence[i]][sentence[i+1]].keys():
				trigrams[sentence[i]][sentence[i+1]][sentence[i+2]] = 1
			else:
				trigrams[sentence[i]][sentence[i+1]][sentence[i+2]] += 1
	return trigrams


def get_fourgrams(tokens):
	fourgrams = {}
	for sentence in tokens:
		for i in range(len(sentence)-3):
			# # print the w1 w2 w3 w4
			# w = sentence[i:i+4]
			# print(w)
			if sentence[i] not in fourgrams.keys():
				fourgrams[sentence[i]] = {}
			if sentence[i+1] not in fourgrams[sentence[i]].keys():
				fourgrams[sentence[i]][sentence[i+1]] = {}
			if sentence[i+2] not in fourgrams[sentence[i]][sentence[i+1]].keys():
				fourgrams[sentence[i]][sentence[i+1]][sentence[i+2]] = {}
			if sentence[i+3] not in fourgrams[sentence[i]][sentence[i+1]][sentence[i+2]].keys():
				fourgrams[sentence[i]][sentence[i+1]][sentence[i+2]][sentence[i+3]] = 1
			else:
				fourgrams[sentence[i]][sentence[i+1]][sentence[i+2]][sentence[i+3]] += 1
	return fourgrams


def ngram_dict(tokens):
	return get_unigrams(tokens), get_bigrams(tokens), get_trigrams(tokens), get_fourgrams(tokens)


def get_unigrams(tokens):
	unigrams = {}
	for sentence in tokens:
		for word in sentence:
			if word not in unigrams.keys():
				unigrams[word] = 1
			else:
				unigrams[word] += 1
	return unigrams


def get_bigrams(tokens):
	bigrams = {}
	for sentence in tokens:
		for i in range(len(sentence)-1):
			if sentence[i] not in bigrams.keys():
				bigrams[sentence[i]] = {}
			if sentence[i+1] not in bigrams[sentence[i]].keys():
				bigrams[sentence[i]][sentence[i+1]] = 1
			else:
				bigrams[sentence[i]][sentence[i+1]] += 1
	return bigrams


def get_trigrams(tokens):
	trigrams = {}
	for sentence in tokens:
		for i in range(len(sentence)-2):
			if sentence[i] not in trigrams.keys():
				trigrams[sentence[i]] = {}
			if sentence[i+1] not in trigrams[sentence[i]].keys():
				trigrams[sentence[i]][sentence[i+1]] = {}
			if sentence[i+2] not in trigrams[sentence[i]][sentence[i+1]].keys():
				trigrams[sentence[i]][sentence[i+1]][sentence[i+2]] = 1
			else:
				trigrams[sentence[i]][sentence[i+1]][sentence[i+2]] += 1
	return trigrams


def get_fourgrams(tokens):
	fourgrams = {}
	for sentence in tokens:
		for i in range(len(sentence)-3):
			# # print the w1 w2 w3 w4
			# w = sentence[i:i+4]
			# print(w)
			if sentence[i] not in fourgrams.keys():
				fourgrams[sentence[i]] = {}
			if sentence[i+1] not in fourgrams[sentence[i]].keys():
				fourgrams[sentence[i]][sentence[i+1]] = {}
			if sentence[i+2] not in fourgrams[sentence[i]][sentence[i+1]].keys():
				fourgrams[sentence[i]][sentence[i+1]][sentence[i+2]] = {}
			if sentence[i+3] not in fourgrams[sentence[i]][sentence[i+1]][sentence[i+2]].keys():
				fourgrams[sentence[i]][sentence[i+1]][sentence[i+2]][sentence[i+3]] = 1
			else:
				fourgrams[sentence[i]][sentence[i+1]][sentence[i+2]][sentence[i+3]] += 1
	return fourgrams


def ngram_dict(tokens):
	return get_unigrams(tokens), get_bigrams(tokens), get_trigrams(tokens), get_fourgrams(tokens)


def wittenbell(unigrams, bigrams, trigrams, fourgrams, input, n):
	if (n == 1):
		# print(input)
		if (input[0] in unigrams):
			return unigrams[input[0]]/sum(unigrams.values())
		else:
			# return 1/sum(unigrams.values())
			# return 1/unigrams["<UNK>"]
			return unigrams["<UNK>"] / sum(unigrams.values())
	elif (n == 2):
		# calculate total number of times history i.e. input[0] occured

		if input[0] not in bigrams.keys():
			return unigrams["<UNK>"] / sum(unigrams.values())
		else:
			C = len(bigrams[input[0]])
			N = sum([item[1] for item in bigrams[input[0]].items()])
			l = N/(N+C)
			if (l == 0):
				return wittenbell(unigrams, bigrams, trigrams, fourgrams, input[1:], 1)
			if (input[1] in bigrams[input[0]]):
				return l*(bigrams[input[0]][input[1]]/N) + (1-l)*wittenbell(unigrams, bigrams, trigrams, fourgrams, input[1:], 1)
			else:
				return (1-l)*wittenbell(unigrams, bigrams, trigrams, fourgrams, input[1:], 1)

	elif (n == 3):
		# calculate total number of times history i.e. input[0][1] occured
		if input[0] not in trigrams.keys():
			return unigrams["<UNK>"] / sum(unigrams.values())

		else:
			# check if trigram[input[0]][input[1]] exists
			if input[1] in trigrams[input[0]].keys():
				C = len(trigrams[input[0]][input[1]])
				N = sum([item[1] for item in trigrams[input[0]][input[1]].items()])
				l = N/(N+C)
				if (l == 0):
					wittenbell(unigrams, bigrams, trigrams, fourgrams, input[1:], 2)

				if (input[2] in trigrams[input[0]][input[1]]):
					return l*(trigrams[input[0]][input[1]][input[2]]/N) + (1-l)*wittenbell(unigrams, bigrams, trigrams, fourgrams, input[1:], 2)
				else:
					return (1-l)*wittenbell(unigrams, bigrams, trigrams, fourgrams, input[1:], 2)

			else:
				return wittenbell(unigrams, bigrams, trigrams, fourgrams, input[1:], 2)

	elif (n == 4):
		# calculate total number of times history i.e. input[0][1][2] occured
		if input[0] not in fourgrams.keys():
			return unigrams["<UNK>"] / sum(unigrams.values())
		else:
			# check if fourgram[input[0]][input[1]]] exists

			if input[1] in fourgrams[input[0]].keys():
				# check if fourgram[input[0]][input[1]][input[2]] exists
				if input[2] in fourgrams[input[0]][input[1]].keys():
					C = len(fourgrams[input[0]][input[1]][input[2]])
					N = sum([item[1]
					        for item in fourgrams[input[0]][input[1]][input[2]].items()])
					l = N/(N+C)
					if (input[3] in fourgrams[input[0]][input[1]][input[2]]):
						return l*(fourgrams[input[0]][input[1]][input[2]][input[3]]/N) + (1-l)*wittenbell(unigrams, bigrams, trigrams, fourgrams, input[1:], 3)
					else:
						return (1-l)*wittenbell(unigrams, bigrams, trigrams, fourgrams, input[1:], 3)
				else:
					return wittenbell(unigrams, bigrams, trigrams, fourgrams, input[1:], 3)
			else:
				return wittenbell(unigrams, bigrams, trigrams, fourgrams, input[1:], 3)

# Interpolated Kneser-Ney
# PKN(wi|wi−n+1:i−1) = max(cKN(wi−n+1:i)−d,0)/sigma_v(cKN(wi−n+1:i−1 v) + l(wi−n+1:i−1)PKN(wi|wi−n+2:i−1)
# l(wi-n+1:i-1) = (d*|{wi : C(wi−n+1:i) > 0}|)/sigma_v (C(wi-n+1:i-1 v))


def kneserney(unigrams, bigrams, trigrams, fourgrams, input, n, orderishigh=True):
	if (n == 1):
		if input[0] not in unigrams.keys():
			return unigrams["<UNK>"] / sum(unigrams.values())
		if (orderishigh):
			return unigrams[input[0]]/sum([item[1] for item in unigrams.items()])
		else:
			# Pcont = |{v : C(vw) > 0}| / |{(u',w'):C(u',w')>0}|
			# Pcontnum = num of unique bigrams with input[1]
			# Pcontdenom = total unique bigrams

			Pcontnum = len(
				set([item[0] for item in bigrams.items() if input[0] in item[1].keys()]))
			Pcontdenom = len(bigrams)
			Pcont = Pcontnum/Pcontdenom
			d = 0.5  # from J&M
			return (max(unigrams[input[0]] - d, 0))/sum([item[1] for item in unigrams.items()]) + d*Pcont
	elif (n == 2):
		d = 0.75
		if input[0] not in bigrams.keys():
			return d/unigrams["<UNK"]
		# get lambda
		# l(wi-n+1:i-1) = (d*|{wi : C(wi−n+1:i) > 0}|)/sigma_v (C(wi-n+1:i-1 v) + l(wi−n+2:i−1)PKN(wi|wi−n+2:i−1)
		try:
			l = (d*len([item for item in bigrams[input[0]].items() if item[1] > 0])
			     )/sum([item[1] for item in bigrams[input[0]].items()])
		except:
			return max(bigrams[input[0]][input[1]]-d, 0)/sum([item[1] for item in bigrams[input[0]].items()]) + 0.75*kneserney(unigrams, bigrams, trigrams, fourgrams, input[1:], 1, False)

		if l != 0:
			if input[1] in bigrams[input[0]].keys():
				return max(bigrams[input[0]][input[1]]-d, 0)/sum([item[1] for item in bigrams[input[0]].items()]) + l*kneserney(unigrams, bigrams, trigrams, fourgrams, input[1:], 1, False)
			else:
				return l*kneserney(unigrams, bigrams, trigrams, fourgrams, input[1:], 1, False) + 1/sum([item[1] for item in bigrams[input[0]].items()])
		else:
			return d/unigrams['<UNK>']

	elif (n == 3):
		d = 0.75
		if input[0] not in trigrams.keys() or input[1] not in trigrams[input[0]].keys():
			return d/unigrams['<UNK>']

		# get lambda
		# l(wi-n+1:i-1) = (d*|{wi : C(wi−n+1:i) > 0}|)/sigma_v (C(wi-n+1:i-1 v) + l(wi−n+2:i−1)PKN(wi|wi−n+2:i−1)
		try:
			l = (d*len([item for item in trigrams[input[0]][input[1]].items() if item[1] > 0])
			     )/sum([item[1] for item in trigrams[input[0]][input[1]].items()])
		except:
			return d*kneserney(unigrams, bigrams, trigrams, fourgrams, input[1:], 2, False)

		if l == 0:
			return d/unigrams['<UNK>']
		if input[1] in trigrams[input[0]].keys():
			if input[2] in trigrams[input[0]][input[1]].keys():
				return max(trigrams[input[0]][input[1]][input[2]]-d, 0)/sum([item[1] for item in trigrams[input[0]][input[1]].items()]) + l*kneserney(unigrams, bigrams, trigrams, fourgrams, input[1:], 2, False)
			else:
				return l*kneserney(unigrams, bigrams, trigrams, fourgrams, input[1:], 2, False) + 1/sum([item[1] for item in trigrams[input[0]][input[1]].items()])
		else:
			return l*kneserney(unigrams, bigrams, trigrams, fourgrams, input[1:], 2, False) + 1/sum([item[1] for item in trigrams[input[0]][input[1]].items()])
	elif (n == 4):
		d = 0.75
		if input[0] not in fourgrams.keys() or input[1] not in fourgrams[input[0]].keys() or input[2] not in fourgrams[input[0]][input[1]].keys():
			return d/unigrams['<UNK>']

		# get lambda
		# l(wi-n+1:i-1) = (d*|{wi : C(wi−n+1:i) > 0}|)/sigma_v (C(wi-n+1:i-1 v) + l(wi−n+2:i−1)PKN(wi|wi−n+2:i−1)
		try:
			l = (d*len([item for item in fourgrams[input[0]][input[1]][input[2]].items() if item[1] > 0])
			     )/sum([item[1] for item in fourgrams[input[0]][input[1]][input[2]].items()])
		except:
			return d*kneserney(unigrams, bigrams, trigrams, fourgrams, input[1:], 3, False)

		if l == 0:
			return d/unigrams['<UNK>']

		if input[1] in fourgrams[input[0]].keys():
			if input[2] in fourgrams[input[0]][input[1]].keys():
				if input[3] in fourgrams[input[0]][input[1]][input[2]].keys():
					return max(fourgrams[input[0]][input[1]][input[2]][input[3]]-d, 0)/sum([item[1] for item in fourgrams[input[0]][input[1]][input[2]].items()]) + l*kneserney(unigrams, bigrams, trigrams, fourgrams, input[1:], 3, False)
				else:
					return l*kneserney(unigrams, bigrams, trigrams, fourgrams, input[1:], 3, False) + 1/sum([item[1] for item in fourgrams[input[0]][input[1]][input[2]].items()])
			else:
				return l*kneserney(unigrams, bigrams, trigrams, fourgrams, input[1:], 3, False) + 1/sum([item[1] for item in fourgrams[input[0]][input[1]][input[2]].items()])
		else:
			return l*kneserney(unigrams, bigrams, trigrams, fourgrams, input[1:], 3, False) + 1/sum([item[1] for item in fourgrams[input[0]][input[1]][input[2]].items()])



def perplexityScore(prob_array):
    sum = 0
    for p in prob_array:
        if p != 0:
            sum += math.log(p)
        else:
            sum += math.log(1e-10)

    return math.exp(-sum/len(prob_array))


# def perplexity(unigrams, bigrams, trigrams, fourgrams, train, test):
# 	# create file - 2020101074_LM1_train-perplexity.txt
# 	# create file - 2020101074_LM1_test-perplexity.txt
# 	# write avg perplexity and line by line perplexity in them against sentence

# 	f = open("2020101074_LM1_train-perplexity.txt", 'w')
# 	f2 = open("2020101074_LM1_test-perplexity.txt", 'w')
# 	# calculate perplexity for train set

# 	# wb and kn give probabiltiy values
# 	# add log probabilities
# 	# divide by number of words
# 	# take power of -1
# 	# put this number on power of 2
# 	# write this number in file agains the sentence
# 	# calculate average perplexity
# 	# write average perplexity in file

# 	prob_array = []
# 	perplexity_array = []
# 	for sentence in train:
# 		for i in range(len(sentence)-3):
# 			# px =  kneserney(unigrams, bigrams, trigrams, fourgrams, sentence[i:i+4], 4, True)
# 			px = wittenbell(unigrams, bigrams, trigrams, fourgrams, sentence[i:i+4], 4)
# 			# print("For",sentence[i:i+4],'probability is:',px)
# 			prob_array.append(px)
# 		try:
# 			perplexityval = perplexityScore(prob_array)
# 		except:
# 			perplexityval = 0
# 		# append sentence, perplexity
# 		perplexity_array.append([sentence, perplexityval])

# 	# get avg perplexity
# 	avg_perplexity = sum([item[1]
# 	                     for item in perplexity_array])/len(perplexity_array)
# 	# write avg perplexity in file
# 	f.write("Average Perplexity for train set is: "+str(avg_perplexity)+"\n")
# 	# write perplexity for each sentence in file in sorted order
# 	# perplexity_array = sorted(perplexity_array, key=lambda x: x[1])
# 	for item in perplexity_array:
# 		f.write(str(item[1])+" "+' '.join(item[0])+"\n")
# 	f.close()

# 	# print(prob_array)
# 	# print(perplexity_array)

# 	prob_array = []
# 	perplexity_array = []
# 	for sentence in test:
# 		for i in range(len(sentence)-3):
# 			# px =  kneserney(unigrams, bigrams, trigrams, fourgrams, sentence[i:i+4], 4, True)
# 			px = wittenbell(unigrams, bigrams, trigrams, fourgrams, sentence[i:i+4], 4)
# 			# print("For",sentence[i:i+4],'probability is:',px)
# 			prob_array.append(px)
# 		# print("Perplexity for sentence",sentence,"is",math.pow(2,(-1/len(sentence))*p))

# 		try:
# 			perplexityval = perplexityScore(prob_array)
# 		except:
# 			perplexityval = 0
# 		# append sentence, perplexity
# 		perplexity_array.append([sentence, perplexityval])

# 	# get avg perplexity
# 	avg_perplexity = sum([item[1]
# 	                     for item in perplexity_array])/len(perplexity_array)
# 	# write avg perplexity in file
# 	f2.write("Average Perplexity for test set is: "+str(avg_perplexity)+"\n")
# 	# write perplexity for each sentence in file in sorted order
# 	# perplexity_array = sorted(perplexity_array, key=lambda x: x[1])
# 	for item in perplexity_array:
# 		f2.write(str(item[1])+" "+' '.join(item[0])+"\n")
# 	f2.close()

# 	# print(prob_array)
# 	# print(perplexity_array)


# # store arguments
smoothing_type = sys.argv[1]
path_to_corpus = sys.argv[2]
# path_to_corpus = "./corpora/Pride and Prejudice - Jane Austen.txt"
corpora = open(path_to_corpus, 'r')
corpora = [line.rstrip('\n') for line in corpora]
# remove new empty lines
corpora = [line for line in corpora if line != '']
# make a tokenized list of sentences
tokenized_dataset = [tokenize_line(line) for line in corpora]
tokenized_dataset = converttoUNK(tokenized_dataset)

# print(tokenized_dataset)

# create train and test set
random.shuffle(tokenized_dataset)
# randomly pick out 1000 sentences from tokenized dataset and store in test set
test_set = random.sample(tokenized_dataset, 1000)
# print(test_set)
# remove test set from tokenized dataset
tokenized_dataset = [i for i in tokenized_dataset if i not in test_set]

# print(test_set)


unigrams, bigrams, trigrams, fourgrams = ngram_dict(tokenized_dataset)

# perplexity(unigrams, bigrams, trigrams, fourgrams, tokenized_dataset, test_set)


# IO
# take input sentence
sentence = input("Enter sentence: ")
# print(sentence)
sentence = tokenize_line(sentence)
# print(sentence)

# convert tokens of sentence into <UNK> if they don't exist in corpora
for i in sentence:
    if i not in unigrams:
        sentence[sentence.index(i)] = "<UNK>"
# print(sentence)


# calculate probability of sentence
p = 0
for i in range(len(sentence)-3):
	if(smoothing_type == "w"):
		px =  wittenbell(unigrams, bigrams, trigrams, fourgrams, sentence[i:i+4], 4)
		print("For",sentence[i:i+4],'probability is:',px)
		p += math.log(px)
	else:
		px = wittenbell(unigrams, bigrams, trigrams, fourgrams, sentence[i:i+4], 4)
		print("For",sentence[i:i+4],'probability is:',px)
		p += math.log(px)


print("Probability of sentence: ", math.exp(p))