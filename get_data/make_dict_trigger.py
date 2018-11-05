
list_trigger = []
result = []
with open('list_trigger.txt','r') as f:
    for line in f:
        for word in line.split('|'):
            word = word.lower()
            list_trigger.append(word)


for word in list_trigger:
  if word not in result:
    result.append(word)

f = open("dictionary.txt", 'w')
for word in result:
  if len(word.split()) == 1:
    f.write(word+"\n")
