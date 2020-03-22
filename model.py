# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

dataset = pd.read_csv(r"C:\Users\SANTHOSH\Desktop\Data_Science\Heroku-Demo\hiring.csv")

X = dataset.iloc[:,:3]

X["Experience"].fillna(0,inplace=True)

X['test_score'].fillna(np.round(X['test_score'].mean(),decimals=2),inplace=True)

def convert_to_int(word):
    word_dict = {'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,
                 'eight':8,'nine':9,'ten':10,'eleven':11,'twelve':12,'zero':0,
                 0:0}
    return word_dict[word]

X['Experience'] = X['Experience'].apply(lambda x:convert_to_int(x))

y = dataset['Salary']

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X,y)

pickle.dump(regressor,open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
print(model.predict([[10,4,8]]))