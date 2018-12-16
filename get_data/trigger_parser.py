import xml.etree.ElementTree as ET
import os

def get_file_path():
    file_path = []
    data_path = '../../ace_2005_td_v7/data/English'
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
    fo = open('trigger_event_data.txt','w')
    ft = open('list_trigger.txt', 'w')
    for file in file_path:
        try:
            tree = ET.parse(file)
            root = tree.getroot()
            for event in root.iter('event'):
                for event_mention in event.iter('event_mention'):
                    ldc_scope = event_mention.find('ldc_scope')
                    content = ldc_scope.find('charseq').text
                    content = content.replace('\n',' ')
                    anchor = event_mention.find('anchor')
                    trigger = anchor.find('charseq').text
                    trigger = trigger.lower().replace('\n',' ')
                    if len(trigger.split()) == 1:
                        if ((trigger != "it") and(trigger != "its") and (trigger != "this") and (trigger != "one") and (trigger != "them") and (trigger != "q&a") and (trigger != "it") and (trigger != "ex") and (trigger != "when") and (trigger != "out") and (trigger != "that") and (trigger != "will") and (trigger != "what") and (trigger != "been")):
                            fo.write(trigger + '|'+ content+'\n')
                            ft.write(trigger + '|')
        except IOError:
            pass
    fo.close()

read_data(get_file_path())
