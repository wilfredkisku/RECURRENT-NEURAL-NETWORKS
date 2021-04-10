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
import matplotlib.pyplot as plt

vocabulary_size = 8000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"
match_tokenizer = RegexpTokenizer("[\w']+")

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
print(tokenized_sentence[14999])
print("The number of sentences read is %d"%(num_sentences))
