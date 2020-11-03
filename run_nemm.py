"""
   CRF-based method for method mention extraction 
   @Author : Hospice Houngbo
"""


import codecs
import numpy as np
import nltk
import pycrfsuite
from bs4 import BeautifulSoup as bs
from bs4.element import Tag
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate
from sklearn.metrics import f1_score
import re
import random

data_bio = []
def split_on_empty_lines(s):
	myarray = s.split("\n\n")
	for e in myarray:
		#print(e)
		l = e.split("\n")
		l = [(e.split("\t")[0],e.split("\t")[1])  for e in l]
		data_bio.append(l)
	return data_bio

docs =[]
with open('methods_mention.txt') as f:
	s = f.read()
	docs = split_on_empty_lines(s)
	#print(docs)
# 
data = []
try:
	for i, doc in enumerate(docs):
	 
		# Obtain the list of tokens in the document
		tokens = [t for t, label in doc]
		#print(tokens)
  
		# Perform POS tagging
		tagged = nltk.pos_tag(tokens)
		#print(tagged)
		# Take the word, POS tag, and its label
		data.append([(w, pos, label) for (w, label), (word, pos) in zip(doc, tagged)])
except:
	   None

def method_word(word):
    return word in ["method", "technique", "approach", "techniques"]
def isNP(sent, i):
    #word = sent[i][0]

    postag = sent[i][1]
    if postag in ["NN","NNS","NNP"] :
        return True
    return False
def isAdj(sent, i):
    #word = sent[i][0]

    postag = sent[i][1]
    if postag in ["JJ","JJS","JJR"] :
        return True
    return False
def contNN(sent,i):
    postag1 = sent[i-1][1]

    return postag1 in ["NN","NNS","NNP"] 
def contNNPlus(sent,i):
    postag1 = sent[i+1][1]
    return postag1 in ["NN","NNS","NNP"] 


def word2features(sent, i):
    word = sent[i][0]

    postag = sent[i][1]
    #print(postag)
    
    features = {
        'bias': 1.0,
        'word': word,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'word.ismethod()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
        'method_word' : method_word(word),
        'isNP': isNP(sent, i),
        "isAdj": isAdj(sent, i)
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        #print(postag1)
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
            #"-1:contNN": contNN(sent, i)
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
            "+1:contNNPlus": contNNPlus(sent, i)
        })
    else:
        features['EOS'] = True

    return features

# A function for extracting features in documents
def extract_features(doc):
    return [word2features(doc, i) for i in range(len(doc))]

# A function fo generating the list of labels for each document
def get_labels(doc):
    return [label for (token, postag, label) in doc]

X = [extract_features(doc) for doc in data]
y = [get_labels(doc) for doc in data]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

trainer = pycrfsuite.Trainer(verbose=True)

# Submit training data to the trainer
for xseq, yseq in zip(X_train, y_train):
    trainer.append(xseq, yseq)

# Set the parameters of the model
trainer.set_params({
    # coefficient for L1 penalty
    'c1': 0.1,

    # coefficient for L2 penalty
    'c2': 0.01,  

    # maximum number of iterations
    'max_iterations': 200,

    # whether to include transitions that
    # are possible, but not observed
    'feature.possible_transitions': True
})

# Provide a file name as a parameter to the train function, such that
# the model will be saved to the file when training is finished
trainer.train('crf.model')

# Generate predictions
tagger = pycrfsuite.Tagger()
tagger.open('crf.model')
random.seed(10)

y_pred = [tagger.tag(xseq) for xseq in X_test]
#print(y_test)

labels = {"B-method": 2, "I-method": 1,"O": 0, "": 0 }

# Convert the sequences of tags into a 1-dimensional array
#print(labels)
truths = np.array([labels[tag] for row in y_test for tag in row])
#print(truths)

try:
    predictions = np.array([labels[tag] for row in y_pred for tag in row])
except:
    NotImplemented
#Print out the classification report
print(classification_report(
        truths, predictions,
        target_names=["O", "I-method", "B-method"]))
print(f1_score(truths, predictions, average=None))
