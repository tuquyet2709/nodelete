import re
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
import datetime
import pandas as pd
import time
import sys
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

def time_diff_str(t1, t2):
  #tinh thoi gian dung de son sanh, t2 phai lon hon t1
  """
  Calculates time durations.
  """
  if (t2 < t1):
    return "error"
  diff = t2 - t1
  mins = int(diff / 60)
  secs = round(diff % 60, 2)
  return str(mins) + " mins and " + str(secs) + " seconds"


def clean_str_vn(string):
  #ham nay hau nhu khong su dung
  """
  Tokenization/string cleaning for all datasets except for SST.
  """
  string = re.sub(r"[~`@#$%^&*-+]", " ", string)
  def sharp(str):
    b = re.sub('\s[A-Za-z]\s\.', ' .', ' '+str)
    while (b.find('. . ')>=0): b = re.sub(r'\.\s\.\s', '. ', b)
    b = re.sub(r'\s\.\s', ' # ', b)
    return b
  string = sharp(string)
  string = re.sub(r" : ", ":", string)
  string = re.sub(r",", " , ", string)
  string = re.sub(r"!", " ! ", string)
  string = re.sub(r"\(", " ( ", string)
  string = re.sub(r"\)", " ) ", string)
  string = re.sub(r"\?", "", string)
  string = re.sub(r"\s{2,}", " ", string)
  return string.strip().lower()

def list_words(mes):
  #ham nay khong co tac dung gi ca
  words = mes.lower().split()
  return " ".join(words)

def load_data(filename):
  #ham lay data tu train.txt hoac test.txt, tra ve 1 data frame tu do, 2 cot : "label", "text"
  res = []
  col1 = []; col2 = []

  with open(filename, 'r') as f:
      for line in f:
        if line != "\n":
          label, p, text = line.split(" ", 2) #chia ra thanh 3 phan "EVENT"(hoac "NEVENT"); ":"; "dong text"
          col1.append(label)
          col2.append(text)

      d = {"label":col1, "text": col2} #tao 1 bien kieu dictionary d mang 2 khoa la 2 mang
      train = pd.DataFrame(d) #su dung cai nay co the tao nhu 1 database, co dot label va text
  return train

def create_model():
    vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_df=0.7, min_df=2, max_features=1000)

    print "Load data..."
    train = load_data('general_data/train.txt')

    print "Data dimensions:", train.shape #kich thuoc cua dataframe (6528, 2)
    print "List features:", train.columns.values #ten cac cot
    print "First review:", train["label"][0], "|", train["text"][0]

    train_text = train["text"].values #tra ve cac doan van ban duoi dang 1 numpy
    vectorizer.fit(train_text)
    #train_text bay gio la 1 numpy chua cac cau, fit no vao tuc ra cau duoc chia ra lam cac tu mot, cac tu do phai
    #co tan suat tu 0.2 den 0.7. day no vao cai vectorizer
    X_train = vectorizer.transform(train_text) #chuyen doi ve kieu term-document matrix.
    X_train = X_train.toarray()
    y_train = train["label"] #'EVENT' hoac 'NEVENT'
    joblib.dump(vectorizer, 'model/vectorizer.pkl')
    fit1(X_train, y_train)
    print "Done"

def fit1(X_train,y_train):
    svm = SVC(kernel='rbf', C=1000)
    svm.fit(X_train, y_train)
    joblib.dump(svm, 'model/svm.pkl')
