import re
import numpy as np

def make_list_5_word(string):
    #a[1][2] de lay phan tu giua
    wordlist = re.sub("[^\w]", " ",string).split()
    if (len(wordlist)) <= 5:
        wordlist = np.asarray(wordlist)
        return wordlist
    result = []
    for idx in range(len(wordlist)-4):
            result.append(wordlist[idx:idx+5])
    result = np.asarray(result)
    return result

# if __name__=="__main__":



