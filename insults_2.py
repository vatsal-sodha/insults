import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import svm
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
insults = df_train['Insult']
insults=insults.tolist()
word_tokens=preProcess(comments)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(comments)

# -----------------------Naiye Bayes--------------------------------------------
# model=MultinomialNB().fit(X_train_counts,df_train['Insult'].tolist())
# # docs_new = ['Fuck you', 'Please, have a seat']
# X_new_counts = count_vect.transform(comments)
# # X_new_tfidf = tfidf_transformer.transform(X_new_counts)
# predicted = model.predict(X_new_counts)
# # train_accuracy=accuracy_score(insults,predicted)

# # print(predicted)
# train_accuracy=accuracy_score(df_train['Insult'].tolist(),predicted)
# X_test_counts = count_vect.transform(test_comments)
# # X_new_tfidf = tfidf_transformer.transform(X_new_counts)
# predicted = model.predict(X_test_counts)
# test_accuracy=accuracy_score(test_labels,predicted)
# # print(test_accuracy)
# print("The training accuracy with Naive Bayes is ",train_accuracy," and the testing accuracy is ",test_accuracy)

# ------------------------------------------------------------------------------

# -----------------------SVM----------------------------------------------------
text_clf_svm = Pipeline([('vect', CountVectorizer()),('clf-svm', SGDClassifier(loss='log', penalty='l2',alpha=1e-3, max_iter=5, random_state=42)),])
parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)],'clf-svm__alpha': (1e-2, 1e-3)}
model = GridSearchCV(text_clf_svm, parameters_svm, n_jobs=-1)
model = model.fit(comments,insults)
predicted = model.predict(comments)
train_accuracy=accuracy_score(insults,predicted)
predicted = model.predict(test_comments)
test_accuracy=accuracy_score(test_labels,predicted)
print("The training accuracy with SVM is ",train_accuracy," and the testing accuracy is ",test_accuracy)
# -------------------------------------------------------------------------------

# model=SGDClassifier().fit(X_train_counts,df_train['Insult'].tolist())
# X_new_counts = count_vect.transform(comments)
# # X_new_tfidf = tfidf_transformer.transform(X_new_counts)
# predicted = model.predict(X_new_counts)
# train_accuracy=accuracy_score(df_train['Insult'].tolist(),predicted)

# X_test_counts = count_vect.transform(test_comments)

# model1=svm.SVC()
# model1.fit(X_train_counts,df_train['Insult'].tolist())
# X_new_counts = count_vect.transform(comments)
# predicted = model1.predict(X_new_counts)

# train_accuracy=accuracy_score(df_train['Insult'].tolist(),predicted)
# X_test_counts = count_vect.transform(test_comments)
# # X_new_tfidf = tfidf_transformer.transform(X_new_counts)
# predicted = model1.predict(X_test_counts)
# # predicted = model.predict(X_test_counts)
# test_accuracy=accuracy_score(test_labels,predicted)
# # print("Naive Bayes accuracy training accuracy is: ",train_accuracy)
# # print("Naive Bayes accuracy test accuracy is:", test_accuracy)
# print("SVM training accuracy is: ",train_accuracy)
# print("SVM test accuracy is:", test_accuracy)