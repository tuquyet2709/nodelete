import nltk

def get_list_pos():
  with open('get_data/dictionary.txt', 'r') as fp:
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
