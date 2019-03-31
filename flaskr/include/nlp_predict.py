# -*- coding: utf-8 -*-
# predict
#
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from janome.tokenizer import Tokenizer
from pandas import Series, DataFrame
import pandas as pd
import pickle
#
class NlpPredict:
    def __init__(self):
        self.params = {}
        self.words=[]
        self.docs=None
        self.answers =[]
        self.get_data()
        self.get_answers()
        self.vecs=None
    #
    def get_token(self, text):
        t = Tokenizer()
        tokens = t.tokenize(text, wakati=True)  # 分かち書きする
        word = ""
        for token in tokens:
            word +=token  + " "
        return word
    #
    def get_data(self):
        csv_data = pd.read_csv("flaskr/files/bun_data.csv" ,encoding="SHIFT-JIS" )
        self.words=csv_data["in"]
        tokens=[]
        for item in csv_data["in"]:
            #print(item)
            token=self.get_token(item)
            tokens.append(token)
        self.docs = np.array(tokens)
    #
    def get_answers(self):
        csv_data = pd.read_csv("flaskr/files/bun_data.csv" ,encoding="SHIFT-JIS" )
        self.answers=csv_data["out"]
    #
    def get_vectorize(self):
        file_name="flaskr/files/params.pkl"
        self.vectorizer =None
        with open(file_name, 'rb') as f:
            self.vectorizer = pickle.load(f)
        print("load vectorizer OK!!")
        self.vecs= self.vectorizer.transform( self.docs )
        return self.vectorizer
    #
    def predict(self, str):
        instr = self.get_token(str ).strip()
        #print("instr=", instr )
        x= self.vectorizer.transform( [  instr ])
        #print( "x=",x)
        #Cosine類似度（cosine_similarity）の算出
        num_sim=cosine_similarity(x , self.vecs)
        #print(num_sim )
        index = np.argmax( num_sim )
        #
        #print("word=", self.words[index])
        #print()
        #ret= self.words[index]
        ret= self.answers[ index ]
        return ret
