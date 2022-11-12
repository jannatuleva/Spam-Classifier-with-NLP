# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 13:12:32 2022

@author: evana
"""


import pandas as pd

messages = pd.read_csv('E:\\Data\\DS\\NLP\\smsspamcollection\\SMSSpamCollection', encoding="ISO-8859-1", sep = '\t',  names = ['label', 'message'])
               
import nltk              
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
               
ps = PorterStemmer()
wordnet = WordNetLemmatizer()
corpus = []


for i in range(len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [wordnet.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
    
#bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000)
X= cv.fit_transform(corpus).toarray()

y = pd.get_dummies(messages['label'])
y= y.iloc[:,1].values


#train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


#training model using Naive bayes

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)

y_pred = spam_detect_model.predict(X_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)