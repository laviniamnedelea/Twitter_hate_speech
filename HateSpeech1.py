import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix


def get_data(filename):

    df = pd.read_csv(filename, delimiter=',')
    labels = df['label'].values
    tweets = df['tweet'].values


    shuffle_stratified = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    for train_index, test_index in shuffle_stratified.split(tweets, labels):
        #print(train_index, test_index)
        t_train, t_test = tweets[train_index], tweets[test_index]
        labels_train, labels_test = labels[train_index], labels[test_index]
    return t_train, labels_train, t_test, labels_test


t_train, y_train, t_test, y_test = get_data('train.csv')

count_vectorizer = CountVectorizer(lowercase=True, analyzer='word', stop_words='english')

count_vectorizer.fit(t_train)
X_train = count_vectorizer.transform(t_train)
#print('X train:', X_train)
X_test = count_vectorizer.transform(t_test)

model = MultinomialNB(alpha=0.5)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

# tweet = np.array([input("Insert a tweet. The sentiment must be obvious:")])
# predict = model.predict(count_vectorizer.transform(tweet))
# if predict:
#     print("This is hate speech")
# else:
#     print("This is not hate speech")
# print(predict)

print(accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))
print(model.predict_proba(X_test[0]))
print('Rezultatele se pot observa din matricea de confuzie:')
print(confusion_matrix(y_test, predictions))
