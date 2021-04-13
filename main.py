import csv
import itertools 
import operator
import numpy as np
import sys
import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer, WordPunctTokenizer
from nltk.probability import FreqDist
from datetime import datetime
from utils import *
#from sklearn.utils.extmath import softmax
import matplotlib.pyplot as plt

##define the variables 
vocabulary_size = 8000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"
match_tokenizer = RegexpTokenizer("[\w']+")
#################
##Part 1
##read the text source for self-supervision
tokenized_sentence = []
num_sentences = 0
print("Reading CSV File ...")
with open('data/reddit-comments-2015-08.csv','r') as f:
    reader = csv.reader(f, skipinitialspace = True)
    
    fields = next(reader) 
    for row in reader:

        ##different ways to tokeinze the sentence
        #tokens_nltk = word_tokenize(row[0])
        #tokens_split = row[0].split()
        #print("The length of the list with"len(tokens_nltk))
        #print(len(tokens_split))
        #print(len(match_tokenizer.tokenize(row[0])))
        
        ##words and punctuations tokenizer
        sentence = WordPunctTokenizer().tokenize(row[0])
        #print(len(sentence))
        temp = [sentence_start_token]+sentence+[sentence_end_token]
        #print(len(tokenized_sentence))
        #data_analysis = FreqDist(tokenized_sentence)
        #data_analysis.plot(100, cumulative=False)
        tokenized_sentence.append(temp)
        num_sentences += 1

#print("The number of sentences read is %d"%(num_sentences))
#print(tokenized_sentence)

##to get the freqency distribution
all_sentences = []
for x in tokenized_sentence:
    all_sentences += x 

#print(all_sentences)

data_analysis = FreqDist(all_sentences)
#print(data_analysis.items())
vocab = data_analysis.most_common(vocabulary_size-1)
vocab_words = [x[0] for x in vocab]
vocab_words.append(unknown_token)
#print(vocab[-1][0])
#print(len(vocab_words))
#print(vocab_words)

vocab_words_index = dict([(w,i) for i,w in enumerate(vocab_words)])
#print(vocab_words_index)

tokenized_sentence_new = []
for i in tokenized_sentence:
    tokenized_sentence_new.append([w if w in vocab_words_index else unknown_token for w in i])

#print(tokenized_sentence_new)
print("Eamples : %s"%tokenized_sentence_new[14999])

#####################
##Part 2
##create the training dataset

X_train = np.asarray([[vocab_words_index[w] for w in sen[:-1]] for sen in tokenized_sentence_new])
Y_train = np.asarray([[vocab_words_index[w] for w in sen[1:]] for sen in tokenized_sentence_new])

#print(X_train[14999])
#print(Y_train[14999])

####################
##Part3

class RNNnumpy:
    def __init__(self, word_dim, hidden_dim = 100, bptt_truncate = 4):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate

        self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
    
    def softmax(self, x):
        return np.exp(x) / np.exp(x).sum(axis=0)

    def forward_propagation(self, x):
        #where x is the input sentence sampled from the dataset (0 to t-1)
        T = len(x)
        s = np.zeros((T+1, self.hidden_dim))
        #print(s.shape)
        s[-1] = np.zeros(self.hidden_dim)
        #print(s.shape)
        
        o = np.zeros((T, self.word_dim))

        for t in np.arange(T):
            s[t] = np.tanh(self.U[:,x[t]] + self.W.dot(s[t-1]))
            o[t] = self.softmax(self.V.dot(s[t]))
        return [o, s]
    
    def predict(self, x):
        o, s = self.forward_propagation(x)
        return np.argmax(o, axis=1)
    
    def calculate_total_loss(self, x, y):
        L = 0
        for i in np.arange(len(y)):
            o, s = self.forward_propagation(x[i])
            correct_word_predictions = o[np.arange(len(y[i])), y[i]]
            L += -1 * np.sum(np.log(correct_word_predictions))
        return L

    def calculate_loss(self, x, y):
        N = np.sum((len(y_i) for y_i in y))
        return self.calculate_total_loss(x, y)/N

obj = RNNnumpy(vocabulary_size)
o, s = obj.forward_propagation(X_train[10])
print(len(X_train[10]))
print(o.shape)
print(o)

predictions = obj.predict(X_train[10])
print(predictions.shape)
print(predictions)

words = [list(vocab_words_index.keys())[list(vocab_words_index.values()).index(p)] for p in predictions]
print(words)
