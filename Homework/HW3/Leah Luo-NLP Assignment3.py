# CIS 668 Assignment 3
# Student: Leah Luo 
# Date: 04/10/2020

import nltk
import re
from nltk import FreqDist
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from nltk.corpus import PlaintextCorpusReader
from nltk.tokenize import word_tokenize
from collections import defaultdict
from nltk.collocations import *
from nltk.corpus import stopwords

from nltk import *

# Retrieve text file 
textfile = open("/Users/LXIN/Desktop/reviews.txt")
reviewText = textfile.read()
print(reviewText[:20])


# nltk.download('sentence_polarity')
from nltk.corpus import sentence_polarity
import random
sentences = sentence_polarity.sents()

print(len(sentences))
print(sentence_polarity.categories())


#Tokenize
from nltk import tokenize
tokensen = tokenize.sent_tokenize(reviewText)
print(len(tokensen))
print(tokensen[:20])


import random
documents = [(sent, cat) for cat in sentence_polarity.categories() for sent in sentence_polarity.sents(categories=cat)]
print(documents[:2])
random.shuffle(documents)
print(documents[0])


all_words_list = [word for (sent,cat) in documents for word in sent]
print(all_words_list[:10])
print(len(all_words_list))

#Fileters: isalpha() and lower()
wordLower = [w for w in all_words_list if w.isalpha()]
print(len(wordLower))
print(wordLower[:20])

# Remove stopwords 
stopwords = nltk.corpus.stopwords.words('english')
wordRmStop = [w for w in wordLower if not w in stopwords]
print("------------------------------------------------------")
print(len(wordRmStop))
print(wordRmStop[:20])

# word_features 1 (without stopwords) 
all_words = nltk.FreqDist(wordRmStop)
word_items = all_words.most_common(2000)
word_features = [word for (word,count) in word_items]
print(word_features[:20])


# Feature 1
SLpath = 'subjclueslen1-HLTEMNLP05.tff'
def readSubjectivity(path):
	flexicon = open(path, 'r')
	# initialize an empty dictionary
	sldict = { }
	for line in flexicon:
		fields = line.split()   # default is to split on whitespace
		# split each field on the '=' and keep the second part as the value
		strength = fields[0].split("=")[1]
		word = fields[2].split("=")[1]
		posTag = fields[3].split("=")[1]
		stemmed = fields[4].split("=")[1]
		polarity = fields[5].split("=")[1]
		if (stemmed == 'y'):
			isStemmed = True
		else:
			isStemmed = False
		sldict[word] = [strength, posTag, isStemmed, polarity]
	return sldict
SL = readSubjectivity(SLpath)

def SL_features(document, word_features, SL):
	document_words = set(document)
	features = {}
	for word in word_features:
		features['contains({})'.format(word)] = (word in document_words)
	# count variables for the 4 classes of subjectivity
	weakPos = 0
	strongPos = 0
	weakNeg = 0
	strongNeg = 0
	for word in document_words:
		if word in SL:
			strength, posTag, isStemmed, polarity = SL[word]
			if strength == 'weaksubj' and polarity == 'positive':
				weakPos += 1
			if strength == 'strongsubj' and polarity == 'positive':
				strongPos += 1
			if strength == 'weaksubj' and polarity == 'negative':
				weakNeg += 1
			if strength == 'strongsubj' and polarity == 'negative':
				strongNeg += 1
			features['positivecount'] = weakPos + (2 * strongPos)
			features['negativecount'] = weakNeg + (2 * strongNeg)      
	return features
	
SL_featuresets = [(SL_features(d, word_features, SL), c) for (d, c) in documents]
train_set, test_set = SL_featuresets[1000:], SL_featuresets[:1000]
classifier = nltk.NaiveBayesClassifier.train(train_set)
nltk.classify.accuracy(classifier, test_set)
classifier.show_most_informative_features(30)

# Reviews SL
#posSL = []
#negSL = []
#
#for s in tokensen:
#	wordToken = nltk.word_tokenize(s)
#	getFeature = SL_features(wordToken, word_features, SL)
#	if classifier2.classify(getFeature) == 'pos':
#		posSL.append(s)
#	elif classifier2.classify(getFeature) == 'neg':
#		negSL.append(s)
#	
#print("neg -------------------------")
#print(len(negSL))
#print(neg[:5])
#
#print("\npos -------------------------")
#print(len(posSL))
#print(pos[:5])



# Feature 2 (Not Feature)
# Negation Words
# this list of negation words includes some "approximate negators" like hardly and rarely
negationwords = ['no', 'not', 'never', 'none', 'nowhere', 'nothing', 'noone', 'rather', 'hardly', 'scarcely', 
                 'rarely', 'seldom', 'neither', 'nor','ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 
                 'haven','isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']
# Remove stop words 
newstopwords = [word for word in stopwords if word not in negationwords]
stop_words_list = [word for word in wordLower if word not in newstopwords]
len(stop_words_list)

all_words = nltk.FreqDist(stop_words_list)
word_items = all_words.most_common(2000)
word_features = [word for (word,count) in word_items]
print(word_features[:20])

def NOT_features(document, word_features, negationwords):
	features = {}
	for word in word_features:
		features['contains({})'.format(word)] = False
		features['contains(NOT{})'.format(word)] = False
	# go through document words in order
	for i in range(0, len(document)):
		word = document[i]
		if ((i + 1) < len(document)) and ((word in negationwords) or (word.endswith("n't"))):
			i += 1
			features['contains(NOT{})'.format(document[i])] = (document[i] in word_features)
		else:
			features['contains({})'.format(word)] = (word in word_features)
	return features
	
# Not Feature Sets
NOT_featuresets = [(NOT_features(d, word_features, negationwords), c) for (d, c) in documents]
NOT_featuresets[0][0]['contains(NOTlike)']

# Accuracy
train_set3, test_set3 = NOT_featuresets[1000:], NOT_featuresets[:1000]
classifier3 = nltk.NaiveBayesClassifier.train(train_set3)
nltk.classify.accuracy(classifier3, test_set3)

# Most informative features
classifier3.show_most_informative_features(30)

# Reviews NOT 
posNOT = []
negNOT = []

for s in tokensen:
	wordToken = nltk.word_tokenize(was)
	getFeature = NOT_features(wordToken, word_features, negationwords)
	if classifier3.classify(getFeature) == 'pos':
		posNOT.append(s)
	elif classifier3.classify(getFeature) == 'neg':
		negNOT.append(s)
				
print("neg -------------------------")
print(len(negNOT))
print(negNOT[:5])

print("\npos -------------------------")
print(len(posNOT))
print(posNOT[:5])


# pos into file 
posFile = open('/Users/LXIN/Desktop/posNOT.txt', 'w')
for r in posNOT:
	posFile.write(r + '\n')
posFile.close()

# neg into file 
negFile = open('/Users/LXIN/Desktop/negNOT.txt', 'w')
for r in negNOT:
	negFile.write(r + '\n')
negFile.close()

# Build the reference and test lists from the classifier on the test set:
reflist = []
testlist = []
for (features, label) in test_set:
	reflist.append(label)
	testlist.append(classifier.classify(features))
	
reflist[:30] 
testlist[:30]

reffemale = set([i for i,label in enumerate(reflist) if label == 'pos']) 
refmale = set([i for i,label in enumerate(reflist) if label == 'neg'])

testfemale = set([i for i,label in enumerate(testlist) if label == 'pos']) 
testmale = set([i for i,label in enumerate(testlist) if label == 'neg'])

from nltk.metrics import *

# compute precision, recall and F-measure for each label
def printmeasures(label, refset, testset):
	print(label, 'precision:', precision(refset, testset))
	print(label, 'recall:', recall(refset, testset)) 
	print(label, 'F-measure:', f_measure(refset, testset))

printmeasures('pos', reffemale, testfemale) 
print("-------------------------------------")
printmeasures('neg', refmale, testmale)


import pandas as pd
import os
col = ['Positive', 'Negative']
define = pd.DataFrame(columns = col)

define['Positive'] = posNOT[:100]
define['Negative'] = negNOT[:100]

file = "/Users/LXIN/Desktop/table.csv"

if not os.path.isfile(file):
	define.to_csv(file, header = True, index = False, encoding = 'utf-8')
print(define)