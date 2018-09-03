import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
# from nltk.stem.lancaster import LancasterStemmer
def preProcess(comments):
    length=len(comments)
    tokenized_data=[]
    unigrams={}
    for i in range(0,length):
#         Converting all text into lowercase
        temp=comments[i].lower()
        comments[i]=temp
#         Removing Punctuation
        temp=comments[i].strip('"')
        # print(temp)
        comments[i]=temp
#     Tokenization
        word_tokens = word_tokenize(comments[i])
        for token in word_tokens:
            if token not in unigrams:
                unigrams[token]=1
            else:
                unigrams[token]+=1
#         tokenized_data.append(word_tokens)
        # i=i+1	
    return unigrams

# def stemming(comments):
#     length=len(comments)
#     stemmed_Data=[]
#     tokenized_data=[]
#     for i in range(0,length):
#         word_tokens = word_tokenize(comments[i])
#         tokenized_data=[]
#         for token in word_tokens:
#             ls=LancasterStemmer()
#             print(token)
#             temp=ls.stem(str(token))
#             tokenized_data.append(temp)  
#         stemmed_Data.append(tokenized_data)
#     return stemmed_Data


train=pd.read_csv("data/train.csv")
test=pd.read_csv("data/test.csv")
test_with_solutions=pd.read_csv("data/test_with_solutions.csv")
df_train=train.drop('Date',1)
df_test=test.drop('Date',1)
test_labels=test_with_solutions['Insult']
test_comments=test_with_solutions['Comment']
# print(df_test.sample())
comments=df_train['Comment']
comments=comments.tolist()
# print(stemming(comments))
# print(comments)
word_tokens=preProcess(comments)
# print(word_tokens)

# classifier=nltk.NaiveBayesClassifier.train(word_tokens)
count_vect = CountVectorizer()
# comments=ls.stem(comments)
X_train_counts = count_vect.fit_transform(comments)
# print(X_train_counts)
# count_vect.vocabulary_.get(u'hollywood')
model1=LogisticRegression()
model1.fit(X_train_counts,df_train['Insult'].tolist())
# model=MultinomialNB().fit(X_train_counts,df_train['Insult'].tolist())

# docs_new = ['Fuck you', 'Please, have a seat']
X_new_counts = count_vect.transform(comments)
# X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = model1.predict(X_new_counts)
# predicted = model.predict(X_new_counts)

# print(predicted)
train_accuracy=accuracy_score(df_train['Insult'].tolist(),predicted)

X_test_counts = count_vect.transform(test_comments)
# X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = model1.predict(X_test_counts)
# predicted = model.predict(X_test_counts)
test_accuracy=accuracy_score(test_labels,predicted)
# print("Naive Bayes accuracy training accuracy is: ",train_accuracy)
# print("Naive Bayes accuracy test accuracy is:", test_accuracy)
print("Logistic Regression training accuracy is: ",train_accuracy)
print("Logistic Regression test accuracy is:", test_accuracy)

# for doc, category in zip(docs_new, predicted):
#     print('%r => %s' % (doc, twenty_train.target_names[category]))
# # print(len(word_tokens))