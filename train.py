# encoding: utf-8
# 2019/03/15 16:28: TfidfVectorizer save
#

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from janome.tokenizer import Tokenizer
import pickle
from pandas import Series, DataFrame
import pandas as pd

#
def get_token(text):
    t = Tokenizer()
    tokens = t.tokenize(text, wakati=True)  # 分かち書きする
    word = ""
    for token in tokens:
        word +=token  + " "
    return word

#
#csv_data = pd.read_csv("bun_data.csv" ,encoding="SHIFT-JIS" )
csv_data = pd.read_csv("flaskr/files/bun_data.csv" ,encoding="SHIFT-JIS" )
#df= pd.read_csv("japanese.csv,encoding="SHIFT-JIS"")

#print(csv_data.head() )

tokens=[]
for item in csv_data["in"]:
    #print(item)
    token=get_token(item)
    tokens.append(token)
#print(len(csv_data.columns) )
#print(tokens )
#quit()

#print(tokens )
#quit()
docs = np.array(tokens)

vectorizer = TfidfVectorizer(use_idf=True, token_pattern=u'(?u)\\b\\w+\\b')
print(tokens)
#quit()
vecs = vectorizer.fit_transform(docs )
print("#vecs :")
print(vecs.shape )
##print(vecs[0] )

#save

#file_name="params.pkl"
file_name="flaskr/files/params.pkl"
with open(file_name, 'wb') as f:
    pickle.dump(vectorizer, f)
print("#save vectorizer OK!")
