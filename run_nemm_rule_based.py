"""
   Rule-based method for method mention extraction 
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
# 
import re
import random

with open("methodsentwithkw2annotate.txt", "r") as fpm:
    docs = fpm.readlines()

# 
data = []
for doc in docs:
    
    tokens = [t.replace("<","").replace(">","") for t in doc.split()]
    tagged = nltk.pos_tag(tokens)
    data.append([t for t in tagged])
i=0
k=0
t = 0
for dt in data:
    k+=1
    st = ""
    for d in dt:
        if d[0] in ["method", "approach", "model", "algorithm", "analysis"]:
            #print(d[0])
            st+=d[0]+" "
            tks = []
            tgs = []
        else:
            st+=d[1]+" "
    pattern = re.compile(r"(( NN| JJ| NNP| JJS| JJR)+( method | analysis | algorithm | approach | model ))")
    try:
        #print(re.search(pattern, st))
        res = re.search(pattern, st)
        if res is not None:
            t+=1
    except:
        None
    try:
        regex = r"(( NN| JJ| NNP| JJS| JJR)+( method | analysis | algorithm | approach | model ))"
        rel = re.findall(regex, st)
        #print(rel[0][0])
        result = rel[0][0]
        # if result!="":
        i+=1
    except:
        None
print ("Accuracy: ", i/k)
print("Accuracy : ", t/k)
