from bs4 import BeautifulSoup

def sgm_paser():
    file_path = "../ace_2005_td_v7/data/English/bn/adj/CNN_ENG_20030414_130735.7.sgm"
    fo = open(file_path,"r")
    doc = fo.read()
    soup = BeautifulSoup(doc,"lxml")
    contents = soup.findAll('turn')
    for content in contents:
        str = content.text
        # str = str.split("\n",2)[2]
        str = str.replace("\n"," ")
        read_data(str)

def sgm_paser2(file_path):
    for file in file_path:
        try:
            fo = open(file, "r")
            doc = fo.read()
            soup = BeautifulSoup(doc, "lxml")
            contents = soup.findAll('turn')
            for content in contents:
                str = content.text
                # str = str.split("\n",2)[2]
                str = str.replace("\n", " ")
                read_data(str)
        except IOError:
            pass

def get_file_path():
    file_path = []
    data_path = '../ace_2005_td_v7/data/English/bn'
    f_list = open(data_path + '/FileList', 'r')
    line = f_list.readline()
    line = f_list.readline()
    while line:
        arr = line.split('\t')
        file_name = arr[0] + ".sgm"
        file_path.append(data_path + '/adj/' + file_name)
        line = f_list.readline()
    f_list.close()
    return file_path

def read_data(str):
    str = str.replace(" --","")
    str = str.replace("? ","\n")
    str = str.replace("! ", "")
    str = str.replace(" (ph)","")
    str = str.replace(" (UNINTELLIGIBLE)","")
    str = str.replace("(voice-over): ","")
    str = str.replace("(on camera): ","")
    str = str.replace("\"","")
    str = str.replace("...","")
    str = str.strip()
    print str
    arr_line = str.split('. ')

    fo = open('content.txt','a')
    i = 0
    for line in arr_line:
        line += "\n"
        line = line.replace(".\n","\n")
        fo.write(line)


if __name__ == '__main__':
    sgm_paser2(get_file_path())
    # sgm_paser()
