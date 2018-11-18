# -*- encoding: utf8 -*-
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
from random import randint

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

def get_list_pos_from_sentence(sentence):
  pos = get_pos_from_sentence(sentence)
  list_pos = []
  for i in range(len(pos)):
    list_pos.append(pos[i][1])
  return list_pos

def get_pos_from_sentence(sentence):
  x = nltk.word_tokenize(sentence)
  pos = nltk.pos_tag(x)
  return pos

def short_lists(lists):
  ignore_pos_list = ["DT", "WRB", "WP", "CD", "PRP", "POS", "CC"]
  index = []
  for i in range(len(lists)):
    main_word = lists[i][2]
    if get_pos_from_sentence(main_word)[0][1] in ignore_pos_list:
      index.append(i)
  short_lists = np.delete(lists, index, 0)
  return short_lists

def random_index(lists, i):
  n = len(lists)
  j = randint(0, n-1)
  while j == i:
    j = randint(0, n-1)
  return j

def check_word_in_dict(word, filename):
  with open(filename, 'r') as f:
    data = f.read().replace('\n', ' ')
  list_words = data.split()
  if word in list_words:
    return 1
  else:
    return 0

def load_trigger_data(filename):
  res = []
  check_trigger = []; string5 = []; main_word_pos_list = []; main_word_pos_in_dict = [];
  full_word_pos = []; main_word_in_dict = []

  with open(filename, 'r') as f:
    for line in f:
      if line != "\n":
        trigger, content = line.split("|")
        lists = make_lists_from_string(content)
        lists = short_lists(lists)
        for i in range(len(lists)):
          main_word = lists[i][2]
          if (main_word == trigger):
            string5.append(convert_list_5_to_string5(lists[i]))
            check_trigger.append("1")
            main_word_pos = get_pos_from_sentence(main_word)[0][1]
            main_word_pos_list.append(main_word_pos)
            main_word_pos_in_dict.append(check_word_in_dict(main_word_pos, "get_data/list_pos.txt"))
            main_word_in_dict.append(check_word_in_dict(main_word, "get_data/dictionary.txt"))
            full_word_pos.append(get_list_pos_from_sentence(convert_list_5_to_string5(lists[i])))

            j = random_index(lists, i) #random index
            random_main_word = lists[j][2]

            string5.append(convert_list_5_to_string5(lists[j]))
            check_trigger.append("0")
            random_main_word_pos = get_pos_from_sentence(random_main_word)[0][1]
            main_word_pos_list.append(random_main_word_pos)
            main_word_pos_in_dict.append(check_word_in_dict(random_main_word_pos, "get_data/list_pos.txt"))
            main_word_in_dict.append(check_word_in_dict(random_main_word, "get_data/dictionary.txt"))
            full_word_pos.append(get_list_pos_from_sentence(convert_list_5_to_string5(lists[j])))

    d = {"string5":string5, "check_trigger": check_trigger, "pos": main_word_pos_list, "full_pos": full_word_pos, "pos_in_dict": main_word_pos_in_dict, "in_dict": main_word_in_dict}
    train = pd.DataFrame(d)
    train.pos = pd.Categorical(train.pos)    #change pos to int
    train['pos'] = train.pos.cat.codes
  return train

def train_main():
    t0 = time.time()
    vectorizer = CountVectorizer(max_features = 1000)
    print "Load data..."
    train = load_trigger_data('general_data/train.txt')
    test = load_trigger_data('general_data/test.txt')

    print "Train data dimensions:", train.shape
    print "Test data dimensions:", test.shape
    print "List features: string5, pos, pos_in_dict, in_dict"

    print "Create vector..."
    X_train = create_X(train, vectorizer)
    X_test = create_X(test, vectorizer)

    y_train = train["check_trigger"].values
    y_test = test["check_trigger"].values

    print "Training vector shape:", X_train.shape
    print "Test vector shape:", X_test.shape

    print "-----------------------TRAINING-----------------------"

    clf = SVC(kernel='rbf', C=500)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # print y_pred
    print "Training completed in %s" % (time_diff_str(t0, time.time()))
    print "Accuracy: %0.3f" % accuracy_score(y_test, y_pred)
    print "Confuse matrix: \n", confusion_matrix(y_test, y_pred, labels=["1", "0"])

def load_model(model):
    print('Loading model ...')
    if os.path.isfile(model):
        return joblib.load(model)
    else:
        return None

def predict_input_sentence(mes):
    print "Load model..."
    svm = load_model('model/svm.pkl')
    if svm == None:
        training()
    print "Load vectorizer..."
    vectorizer = load_model('model/vectorizer.pkl')

    string5 = []
    string5.append(mes)
    d2 = {"message": string5}
    #TODO------------------------------------------------------------------------------------
    test = pd.DataFrame(d2)
    test_text = test["message"].values.astype('str')
    test_data_features = vectorizer.transform(test_text)
    test_data_features = test_data_features.toarray()
    # print test_data_features
    s = svm.predict(test_data_features)[0]
    return s

def fit_SVM(X_train,y_train):
    svm = SVC(kernel='rbf', C=1000)
    svm.fit(X_train, y_train)
    joblib.dump(svm, 'model/svm.pkl')

def create_X(mode, vectorizer):
    train_string5 = mode["string5"].values
    string5_vectorizer = vectorizer.fit_transform(train_string5)
    train_features = mode[["pos", "pos_in_dict", "in_dict"]].values
    a = string5_vectorizer.toarray()
    X_train = np.concatenate((a, train_features[:None]), axis=1)
    return X_train

def create_model():
    t0 = time.time()
    vectorizer = CountVectorizer(max_features = 1000)

    print "Load data..."
    train = load_trigger_data('general_data/train.txt')

    X_train = create_X(train, vectorizer)
    print "X_train:"
    print X_train
    y_train = train["check_trigger"].values
    print "y_train:"
    print y_train
    joblib.dump(vectorizer, 'model/vectorizer.pkl')
    fit_SVM(X_train, y_train)
    print "Model has been created, completed in %s" % (time_diff_str(t0, time.time()))

if __name__ == '__main__':
    mode = ' '.join(sys.argv[1:])

    if mode == "train":
        train_main()

    elif mode == "model":
        create_model()

    elif mode == "custom_input":
        mes = raw_input("Input list 5 word: ")
        result = predict_input_sentence(mes)
        print "Result: " + result
    else:
        print "Error argument!"

# print load_trigger_data("get_data/trigger_event_data.txt")
# create_model()
