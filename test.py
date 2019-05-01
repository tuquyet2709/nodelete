# execfile("/home/tuquyet/GR/nodelete/quyet.py")
from __future__ import division
import quyet

def predict(mes):
  vectorizer_pred = quyet.CountVectorizer(max_features = 5)
  svm = quyet.load_model('model/svm.pkl')
  vectorizer = quyet.load_model('model/vectorizer.pkl')

  string5 = []; main_word_pos_list = []; main_word_pos_in_dict = [];
  full_word_pos = []; main_word_in_dict = []

  lists = quyet.make_lists_from_string(mes)
  for i in range(len(lists)):
    main_word = lists[i][2]
    string5.append(quyet.convert_list_5_to_string5(lists[i]))
    main_word_pos = quyet.get_pos_from_sentence(main_word)[0][1]
    main_word_pos_list.append(main_word_pos)
    main_word_pos_in_dict.append(quyet.check_word_in_dict(main_word_pos, "get_data/list_pos.txt"))
    main_word_in_dict.append(quyet.check_word_in_dict(main_word, "get_data/dictionary.txt"))
    full_word_pos.append(quyet.get_list_pos_from_sentence(quyet.convert_list_5_to_string5(lists[i])))

  d2 = {"string5":string5, "pos": main_word_pos_list, "full_pos": full_word_pos, "pos_in_dict": main_word_pos_in_dict, "in_dict": main_word_in_dict}

  pred = quyet.pd.DataFrame(d2)
  pred.pos = quyet.pd.Categorical(pred.pos)
  pred['pos'] = pred.pos.cat.codes
  X = quyet.create_X(pred, vectorizer_pred)
  pred_list = []
  for i in range(len(X)):
    s = svm.predict([X[i]])
    if (s[0] == "1"):
      pred_list.append(lists[i][2])
  result = quyet.post_processing(mes, pred_list) #hau xu li
  return result

def test():
  trigger_list = []; content_list = []
  with open('general_data/test.txt', 'r') as f:
    for line in f:
      if line != "\n":
        trigger, content = line.split("|")
        if len(content.split()) < 6:
          continue
        else:
          trigger_list.append(trigger)
          content_list.append(content)

  n = len(content_list)
  a = 0
  equal = 0
  for i in range(n):
    pred = predict(content_list[i])
    a = a + len(pred)
    if trigger_list[i] in pred:
      equal = equal +1

  print n
  print a
  print equal

  p = equal/n
  r = equal/a
  f = (2*p*r)/(p+r)
  print "p = " + str(p)
  print "r = " + str(r)
  print "f = " + str(f)
test()
