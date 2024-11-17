import numpy as np
import pandas as pd
from scipy import stats
import nltk
from nltk.stem import WordNetLemmatizer 
import re

def get_mean_confidence_interval(data,alpha=0.05):
    sample_mean = np.mean(data)
    simple_std = np.std(data,ddof=1)
    n = len(data)

    t = stats.t.ppf(1-alpha/2,n-1)
    
    tmp = t*(simple_std/np.sqrt(n))
    return {'p-vaule':0.05,'lower':sample_mean-tmp,'upper':sample_mean+tmp}



def oneway_anvoa(data):
    group_sum = [np.sum(i) for i in data]
    group_numbers = [len(i) for i in data]

    #誤差因子
    CM = np.sum(group_sum)**2/np.sum(group_numbers)
    
    #總變異
    SST = np.sum([np.sum([ i**2 for i in g]) for g in data]) - CM
    #總間變異
    SSG = np.sum([np.sum(i)**2/len(i) for i in data]) - CM

    #組內變異
    SSB = SST - SSG

    F = (SSG/(len(group_numbers)-1)) / (SSB/(np.sum(group_numbers)-len(group_numbers)))

    p_value = stats.f.sf(F, len(group_numbers)-1, np.sum(group_numbers)-len(group_numbers))

    return {"F":F,"p-vaule":p_value} 

def convert(sentences,stop_words):
    lemmatizer = WordNetLemmatizer()

    lst_sentences = []
    for s in sentences:
        s = s.replace('title','')
        s = s.replace('text','')
        lst = []
        for ss in s.split(" "):
            word = re.search('\w+', ss)
            if word is not None:
                word = word.group()
                word = word.lower()
                word = lemmatizer.lemmatize(word)
                if word not in stop_words:
                    lst.append(word)
        lst_sentences.append(" ".join(lst))
    return " ".join(lst_sentences).split(" ")