# CIS 668 Assignment 3 (Bonus)
# Student: Leah Luo 
# Date: 04/10/2020

import nltk
# Download the sample tweets from the NLTK package
nltk.download('twitter_samples')

nltk.download('punkt')

from nltk.corpus import twitter_samples

# negative_tweets.json: 5000 tweets with negative sentiments
# positive_tweets.json: 5000 tweets with positive sentiments
# tweets.20150430-223406.json: 20000 tweets with no sentiments

positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')
text = twitter_samples.strings('tweets.20150430-223406.json')
tweet_tokens = twitter_samples.tokenized('positive_tweets.json')


# Normalizing the Data
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

from nltk.tag import pos_tag
from nltk.corpus import twitter_samples
tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
print(pos_tag(tweet_tokens[0]))


from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer

# This code imports the WordNetLemmatizer class and initializes it to a variable, lemmatizer
def lemmatize_sentence(tokens):
	lemmatizer = WordNetLemmatizer()
	lemmatized_sentence = []
	for word, tag in pos_tag(tokens):
		if tag.startswith('NN'):
			pos = 'n'
		elif tag.startswith('VB'):
			pos = 'v'
		else:
			pos = 'a'
		lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
	return lemmatized_sentence

print(lemmatize_sentence(tweet_tokens[0]))


# Removing Noise from the Data

# Use regular expressions in Python to search for and remove these items:
# Hyperlinks - All hyperlinks in Twitter are converted to the URL shortener t.co.
# Twitter handles in replies 
# Punctuation and special characters 

import re, string

def remove_noise(tweet_tokens, stop_words = ()):

	cleaned_tokens = []

	for token, tag in pos_tag(tweet_tokens):
		token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
					   '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
		token = re.sub("(@[A-Za-z0-9_]+)","", token)

		if tag.startswith("NN"):
			pos = 'n'
		elif tag.startswith('VB'):
			pos = 'v'
		else:
			pos = 'a'

		lemmatizer = WordNetLemmatizer()
		token = lemmatizer.lemmatize(token, pos)

		if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
			cleaned_tokens.append(token.lower())
	return cleaned_tokens
	

# Remove stopwords 
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

#print(remove_noise(tweet_tokens[0], stop_words))

positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

positive_cleaned_tokens_list = []
negative_cleaned_tokens_list = []

for tokens in positive_tweet_tokens:
	positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

for tokens in negative_tweet_tokens:
	negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))
	
# Determining Word Density
# Take a list of tweets as an argument to provide a list of words in all of the tweet tokens joined
def get_all_words(cleaned_tokens_list):
	for tokens in cleaned_tokens_list:
		for token in tokens:
			yield token

all_pos_words = get_all_words(positive_cleaned_tokens_list)


from nltk import FreqDist

freq_dist_pos = FreqDist(all_pos_words)
print(freq_dist_pos.most_common(10))


# Preparing Data for the Model

# Converting Tokens to a Dictionary
def get_tweets_for_model(cleaned_tokens_list):
	for tweet_tokens in cleaned_tokens_list:
		yield dict([token, True] for token in tweet_tokens)

positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)


# Splitting the Dataset for Training and Testing the Model
import random

positive_dataset = [(tweet_dict, "Positive")
					 for tweet_dict in positive_tokens_for_model]

negative_dataset = [(tweet_dict, "Negative")
					 for tweet_dict in negative_tokens_for_model]

dataset = positive_dataset + negative_dataset

random.shuffle(dataset)

train_data = dataset[:7000]
test_data = dataset[7000:]


# Building and Testing the Model
from nltk import classify
from nltk import NaiveBayesClassifier

classifier = NaiveBayesClassifier.train(train_data)

print("Accuracy is:", classify.accuracy(classifier, test_data))

print(classifier.show_most_informative_features(10))


# Retrieve text file 
textfile = open("/Users/LXIN/Desktop/reviews.txt")
reviewText = textfile.read()
print(reviewText[:20])

#Tokenize
from nltk import tokenize
tokensen = tokenize.sent_tokenize(reviewText)
print(len(tokensen))
print(tokensen[:20])


# Reviews  
posBouns = []
negBouns = []

from nltk.tokenize import word_tokenize

for s in tokensen:
	wordToken = remove_noise(word_tokenize(s))
	
	if classifier.classify(dict([token, True] for token in wordToken)) == 'Positive':
		posBouns.append(s)
	elif classifier.classify(dict([token, True] for token in wordToken)) == 'Negative':
		negBouns.append(s)
				
print("neg -------------------------")
print(len(negBouns))
print(negBouns[:5])

print("\npos -------------------------")
print(len(posBouns))
print(posBouns[:5])


# pos into file 
posFile = open('/Users/LXIN/Desktop/posBonus.txt', 'w')
for r in posBouns:
	posFile.write(r + '\n')
posFile.close()

# neg into file 
negFile = open('/Users/LXIN/Desktop/negBonus.txt', 'w')
for r in negBouns:
	negFile.write(r + '\n')
negFile.close()


import pandas as pd
import os
col = ['Positive', 'Negative']
define = pd.DataFrame(columns = col)

define['Positive'] = posBouns[:100]
define['Negative'] = negBouns[:100]

file = "/Users/LXIN/Desktop/tableBonus.csv"

if not os.path.isfile(file):
	define.to_csv(file, header = True, index = False, encoding = 'utf-8')
print(define)