import csv
import itertools 
import operator
import numpy as np
import sys
from nltk.tokenize import word_tokenize, RegexpTokenizer, WordPunctTokenizer
from datetime import datetime
from utils import *
import matplotlib.pyplot as plt
import re

vocabulary_size = 8000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"
match_tokenizer = RegexpTokenizer("[\w']+")

print("Reading CSV File ...")
with open('data/reddit-comments-2015-08.csv','r') as f:
    reader = csv.reader(f, skipinitialspace = True)
    
    fields = next(reader) 
    for row in reader:
        #tokens_nltk = word_tokenize(row[0])
        #tokens_split = row[0].split()
        #print("The length of the list with"len(tokens_nltk))
        #print(len(tokens_split))
        #print(len(match_tokenizer.tokenize(row[0])))
        sentence = WordPunctTokenizer().tokenize(row[0])
        print(len(sentence))
        tokenized_sentence = [sentence_start_token]+sentence+[sentence_end_token]
        print(len(tokenized_sentence))
        break
print("The number of sentences read is %d"%(reader.line_num - 1))
