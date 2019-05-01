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
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize

from nltk.parse.stanford import StanfordDependencyParser
os.environ['CLASSPATH'] = '/home/tuquyet/GR/stanford-parser-full-2018-10-17/stanford-parser.jar'
os.environ['STANFORD_MODELS'] = '/home/tuquyet/GR/stanford-parser-full-2018-10-17/stanford-parser-3.9.2-models.jar'

def time_diff_str(t1, t2):
  if (t2 < t1):
    return "error"
  diff = t2 - t1
  mins = int(diff / 60)
  secs = round(diff % 60, 2)
  return str(mins) + " mins and " + str(secs) + " seconds"

def make_lists_from_string(string):
  #a[1][2] de lay phan tu giua
  # wordlist = re.sub("[^\w'], [^\w-]", " ",string).split()
  wordlist = re.sub("[^\w-]", " ",string).split()
  # wordlist = re.split(" ", string)
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

def is_number(word):
  if word.isdigit():
    return 1
  else:
    return 0

def is_caps(word):
  if word == "":
    return 0
  if word[0].isupper():
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
  vectorizer = CountVectorizer(max_features = 5)
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

def check_verb(word):
  if word in ["VB", "VBN", "VBD", "VBG", "VBZ", "VBP"]:
    return 1
  else:
    return 0

def check_noun(word):
  if word in ["NN", "NNS", "NNP", "NNPS"]:
    return 1
  else:
    return 0

def check_adj(word):
  if word in ["JJ", "JJR", "JJS"]:
    return 1
  else:
    return 0

def get_front_word(sentence, word):
  list_words = re.sub("[^\w-]", " ",sentence).split()
  # list_words = re.sub("[^\w'], [^\w-]", " ",sentence).split()
  index = list_words.index(word)
  if index == 0:
    return None
  else:
    return list_words[index - 1]

def get_trigger_from_lever(pred_list): #top 1, top 2,...
  top1 = ["killed", "death", "war", "meeting", "died", "attack", "trial"] #>50 times
  top2 = ["fighting", "attacks", "election", "convicted", "arrested", "appeal", "sentence", "elections", "charges", "former"
  , "go", "fire", "former"] #30-50 times
  top3 = ["injured", "dead", "pay", "sentenced", "bombing", "call", "charged", "divorce", "shot", "talks", "die"
  , "kill", "battle", "hit", "fired", "fight", "killing", "strike", "terrorism", "come", "bankruptcy", "combat"] #20-30 times

  top1_result = []
  top2_result = []
  top3_result = []

  for i in range(len(pred_list)):
    if pred_list[i] in top1:
      top1_result.append(pred_list[i])
    elif pred_list[i] in top2:
      top2_result.append(pred_list[i])
    elif pred_list[i] in top3:
      top3_result.append(pred_list[i])

  print "--------------------TOP_TRIGGER--------------------"
  print top1_result
  print top2_result
  print top3_result

  if(len(top1_result) != 0):
    return top1_result
  else:
    if (len(top2_result) != 0):
      return top2_result
    else:
      if (len(top3_result) != 0):
        return top3_result
      else:
        return pred_list

def post_processing(mes, pred_list):
  pos = get_pos_from_sentence(mes)
  result = ""
  pos1 = ""
  pos2 = ""

  if (len(pred_list) == 1):
    result = pred_list[0].split()

  elif (len(pred_list) == 2):
    for i in range(len(pos)):
      if pos[i][0] == pred_list[0]:
        pos1 = pos[i][1]
        print pos1
      if pos[i][0] == pred_list[1]:
        pos2 = pos[i][1]
        print pos2

    if ((check_verb(pos1) == 1) and ((check_noun(pos2) == 1) or (check_adj(pos2) == 1))): #VERB + NOUN or VERB + ADJ
      result = pred_list[1].split()

    elif ((check_verb(pos1) == 1) and (check_verb(pos2) == 1)): #VERB + VERB
      front_word1 = get_front_word(mes, pred_list[0])
      front_word2 = get_front_word(mes, pred_list[1])
      if front_word1 in ["be", "been", "have", "has", "to", "can", "could", "was", "were"]:
        result = pred_list[0].split()
      if front_word2 in ["be", "been", "have", "has", "to", "can", "could", "was", "were"]:
        result = pred_list[1].split()

    elif ((check_noun(pos1) == 1) and (check_verb(pos2) == 1)): #NOUN + VERB
      result = pred_list[0].split()
    else:
      result = get_trigger_from_lever(pred_list)
  else:
    result = get_trigger_from_lever(pred_list)
  return result

def getNER(word, sentence):
  ner = StanfordNERTagger('/home/tuquyet/GR/stanford-ner-2018-10-16/classifiers/english.all.3class.distsim.crf.ser.gz','/home/tuquyet/GR/stanford-ner-2018-10-16/stanford-ner.jar', encoding='utf-8')
  nerList = []
  list_words = word.split()
  tokenized_text = word_tokenize(sentence)
  classified_text = ner.tag(tokenized_text)
  for i in range(len(list_words)):
    for j in range(len(classified_text)):
      if classified_text[j][0] == list_words[i]:
        nerList.append(classified_text[j][1])
  result = False
  if len(nerList) > 0:
    result = all(elem == nerList[0] for elem in nerList) #all element is equal
  if result :
    return nerList[0]
  else:
    for k in range(len(nerList)):
      print nerList[k]

def get_word_from_tree(dep, i):
  root = dep._tree(i).label()
  leaves = dep._tree(i).leaves()
  temp = ' '.join(leaves)
  return temp + " " + root

def get_dependency(node, dep, sentence, temp):
  dep_list = node["deps"]
  if "nsubj" in dep_list.keys():
    for i in range(len(dep_list["nsubj"])):
      j = dep_list["nsubj"][i]
      if j in temp: # da tung xet cay roi
        break
      sub = sentence.split()[j - 1]
      if str(type(dep._tree(j))) != "<type 'unicode'>": #is a tree
        sub = get_word_from_tree(dep, j)
      print "Subject : " + sub
      print getNER(sub, sentence)

  if "dobj" in dep_list.keys():
    for i in range(len(dep_list["dobj"])):
      j = dep_list["dobj"][i]
      if j in temp: # da tung xet cay roi
        break
      argument = sentence.split()[j - 1]
      if str(type(dep._tree(j))) != "<type 'unicode'>": #is a tree
        argument = get_word_from_tree(dep, j)
      print "Argument : " + argument
      print getNER(argument, sentence)

  if "nmod" in dep_list.keys():
    for i in range(len(dep_list["nmod"])):
      j = dep_list["nmod"][i]
      if j in temp: # da tung xet cay roi
        break
      argument = sentence.split()[j - 1]
      if str(type(dep._tree(j))) != "<type 'unicode'>": #is a tree
        argument = get_word_from_tree(dep, j)
      print "Argument : " + argument
      print getNER(argument, sentence)

  if "nmod:tmod" in dep_list.keys():
    for i in range(len(dep_list["nmod:tmod"])):
      j = dep_list["nmod:tmod"][i]
      if j in temp: # da tung xet cay roi
        break
      argument = sentence.split()[j - 1]
      if str(type(dep._tree(j))) != "<type 'unicode'>": #is a tree
        argument = get_word_from_tree(dep, j)
      print "Argument : " + argument
      print getNER(argument, sentence)

def dependency_parser(sentence, result):
  dep_parser=StanfordDependencyParser(model_path="/home/tuquyet/GR/stanford-english-corenlp-2018-10-05-models/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
  sub = ""; argument = "";
  dep = dep_parser.raw_parse(sentence).next()
  dep.tree().pretty_print()
  print list(dep.triples())

  for i in range(len(result)):
    word = result[i]
    print "Trigger : " + word
    if word in sentence:
      index = sentence.split().index(word) + 1
      temp = []
      while True:
        temp.append(index)
        node = dep.get_by_address(index)
        get_dependency(node, dep, sentence, temp)
        index = dep._hd(index)
        if index == 0:
          break
    else:
      print "Error!"

def predict_input_sentence(mes):
  vectorizer_pred = CountVectorizer(max_features = 5)
  print "Load model..."
  svm = load_model('model/svm.pkl')
  if svm == None:
      training()
  print "Load vectorizer..."
  vectorizer = load_model('model/vectorizer.pkl')

  string5 = []; main_word_pos_list = []; main_word_pos_in_dict = [];
  full_word_pos = []; main_word_in_dict = []

  lists = make_lists_from_string(mes)
  # lists = short_lists(lists)
  for i in range(len(lists)):
    main_word = lists[i][2]
    string5.append(convert_list_5_to_string5(lists[i]))
    main_word_pos = get_pos_from_sentence(main_word)[0][1]
    main_word_pos_list.append(main_word_pos)
    main_word_pos_in_dict.append(check_word_in_dict(main_word_pos, "get_data/list_pos.txt"))
    main_word_in_dict.append(check_word_in_dict(main_word, "get_data/dictionary.txt"))
    full_word_pos.append(get_list_pos_from_sentence(convert_list_5_to_string5(lists[i])))

  d2 = {"string5":string5, "pos": main_word_pos_list, "full_pos": full_word_pos, "pos_in_dict": main_word_pos_in_dict, "in_dict": main_word_in_dict}

  pred = pd.DataFrame(d2)
  pred.pos = pd.Categorical(pred.pos)
  pred['pos'] = pred.pos.cat.codes
  X = create_X(pred, vectorizer_pred)
  pred_list = []
  for i in range(len(X)):
    print "---------------------------------------------"
    print lists[i]
    print X[i]
    s = svm.predict([X[i]])
    print "=> "+ s[0]
    if (s[0] == "1"):
      pred_list.append(lists[i][2])
  print "List words can be a trigger :"
  print pred_list
  result = post_processing(mes, pred_list) #hau xu li
  print "================================================="
  print "TRIGGER : "
  print result
  dependency_parser(mes, result)
  print "================================================="

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
    vectorizer = CountVectorizer(max_features = 5)

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
      mes = raw_input("Input a sentence: ")
      predict_input_sentence(mes)
    else:
      print "Error argument!"

# print load_trigger_data("get_data/trigger_event_data.txt")
