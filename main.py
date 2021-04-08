import csv
import itertools 
import operator
import numpy as np
import sys
import nltk
from datetime import datetime
from utils import *
import matplotlib.pyplot as plt

vocabulary_size = 8000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

