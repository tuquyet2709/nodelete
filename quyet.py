import re
import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
import datetime
import pandas as pd
import time
import sys
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import nltk

def time_diff_str(t1, t2):
  if (t2 < t1):
    return "error"
  diff = t2 - t1
  mins = int(diff / 60)
  secs = round(diff % 60, 2)
  return str(mins) + " mins and " + str(secs) + " seconds"

def make_lists_from_string(string):
  #a[1][2] de lay phan tu giua
  wordlist = re.sub("[^\w]", " ",string).split()
  wordlist = ["", ""] + wordlist + ["", ""]
  result = []
  for idx in range(2, len(wordlist)-2):
    result.append(wordlist[idx-2:idx+3])
  result = np.asarray(result)
  return result

def convert_list_5_to_string5(list5):
  string = ""
  for i in range(len(list5)):
    if (list5[i] != ""):
      string = string + list5[i]
      if i != 4:
        string = string + " "
  return string

def get_pos_from_sentence(sentence):
  x = nltk.word_tokenize(sentence)
  pos = nltk.pos_tag(x)
  return pos

def load_trigger_data(filename):
  res = []
  string5 = []; check_trigger = []
  with open(filename, 'r') as f:
    for line in f:
      if line != "\n":
        trigger, content = line.split("|")
        lists = make_lists_from_string(content)
        for i in range(len(lists)):
          if (lists[i][2] == trigger):
            string5.append(convert_list_5_to_string5(lists[i]))
            check_trigger.append("trigger")
            # string5.append(convert_list_5_to_string5(lists[2]))
            # check_trigger.append("not trigger")
          # else:
          #   check_trigger.append("not trigger")
    d = {"string5":string5, "check_trigger": check_trigger}
    train = pd.DataFrame(d)
  return train


def train_main():
    vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_df=0.7, min_df=2, max_features=None)
    train = load_trigger_data('get_data/trigger_event_data.txt')

    print "Data dimensions:", train.shape
    print "List features:", train.columns.values

    train_string5 = train["string5"].values

    vectorizer.fit(train_string5)
    X_train = vectorizer.transform(train_string5)
    X_train = X_train.toarray()
    y_train = train["check_trigger"]
    print X_train

    print "---------------------------"
    print "Training"
    print "---------------------------"
    names = ["RBF SVC"]
    t0 = time.time()
    # iterate over classifiers

    clf = SVC(kernel='rbf', C=500)
    clf.fit(X_train, y_train)
    # print y_pred

    print " %s - Converting completed %s" % (datetime.datetime.now(), time_diff_str(t0, time.time()))

def load_model(model):
    print('Loading model ...')
    if os.path.isfile(model):
        return joblib.load(model)
    else:
        return None

def predict_ex(mes):
    svm = load_model('model/svm.pkl')
    if svm == None:
        training()
    vectorizer = load_model('model/vectorizer.pkl')
    svm = load_model('model/svm.pkl')
    print "---------------------------"
    print "Training"
    print "---------------------------"

    # test_message = list_words(test_message) # lam thanh chu thuong
    clean_test_reviews = []
    clean_test_reviews.append(mes)
    d2 = {"message": clean_test_reviews}
    test = pd.DataFrame(d2)
    test_text = test["message"].values.astype('str')
    test_data_features = vectorizer.transform(test_text)
    test_data_features = test_data_features.toarray()
    # print test_data_features
    s = svm.predict(test_data_features)[0]
    return s

def fit1(X_train,y_train):
    svm = SVC(kernel='rbf', C=1000)
    svm.fit(X_train, y_train)
    joblib.dump(svm, 'model/svm.pkl')

def create_model():
    vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_df=0.7, min_df=2, max_features=1000)

    print "Load data..."
    train = load_trigger_data('get_data/trigger_event_data.txt')

    print "Data dimensions:", train.shape
    print "List features:", train.columns.values

    train_string5 = train["string5"].values
    vectorizer.fit(train_string5)
    X_train = vectorizer.transform(train_string5)
    X_train = X_train.toarray()
    y_train = train["check_trigger"]
    joblib.dump(vectorizer, 'model/vectorizer.pkl')
    fit1(X_train, y_train)
    print "Done"

# if __name__ == '__main__':
#     mode = ' '.join(sys.argv[1:])

#     if mode == "train":
#         train_main()

#     elif mode == "model":
#         create_model()

#     elif mode == "custom_input":
#         mes = raw_input("Custom input: ")
#         kq = predict_ex(mes)
#         print "Result: " + kq
#     else:
        # print "Error argument!"

# print load_trigger_data('get_data/trigger_event_data.txt')
# print make_list_from_string("i have a question now")
