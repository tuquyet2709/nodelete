
def freq(str):
    str = str.split("|")
    str2 = []

    for i in str:
        if i not in str2:
            str2.append(i)

    f = open("top.txt", 'w')
    for i in range(0, len(str2)):

        f.write("{} {}\n".format(str2[i],str.count(str2[i])))
        # print(str2[i],str.count(str2[i]))


with open('list_trigger.txt','r') as f:
    data = f.read()
    freq(data)
