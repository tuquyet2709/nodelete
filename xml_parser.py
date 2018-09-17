import xml.etree.ElementTree as ET
import os

def get_file_path():
    file_path = []
    data_path = '../ace_2005_td_v7/data/English'
    list_dir = os.listdir(data_path)
    for dir in list_dir:
        f_list = open(data_path+'/'+dir+'/FileList','r')
        line = f_list.readline()
        line = f_list.readline()
        while line:
            arr = line.split('\t')
            file_name = arr[0] + ".apf.xml"
            file_path.append(data_path+'/'+dir+'/adj/'+file_name)
            line = f_list.readline()
        f_list.close()
    return file_path

def read_data(file_path):
    fo = open('event_data.txt','w')
    for file in file_path:
        try:
            tree = ET.parse(file)
            root = tree.getroot()
            for event in root.iter('event'):
                for event_mention in event.iter('event_mention'):
                    ldc_scope = event_mention.find('ldc_scope')
                    content = ldc_scope.find('charseq').text
                    content = content.replace('\n',' ')
                    fo.write(content+'\n')
        except IOError:
            pass
    fo.close()

read_data(get_file_path())
