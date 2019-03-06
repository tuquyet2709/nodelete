import nltk

def get_pos_from_sentence(sentence):
  x = nltk.word_tokenize(sentence)
  pos = nltk.pos_tag(x)
  return pos

def get_list_pos():
  with open('dictionary.txt', 'r') as fp:
   line = fp.readline()
   cnt = 1
   listo = []
   while line:
      a = get_pos_from_sentence(line)
      print a
      if a[0][1] not in listo:
        listo.append(a[0][1])
      line = fp.readline()
      cnt += 1
   print listo

get_list_pos()
