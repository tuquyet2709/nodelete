import xml.etree.ElementTree as ET
import os

LIFE = {"count" : 0, "be_born": 0, "marry": 0, "divorce" : 0,"injure" : 0, "die": 0}
MOVEMENT = {"count" : 0, "transport": 0}
TRANSACTION = {"count" : 0, "transfer_ownership": 0, "transfer_money": 0}
BUSINESS = {"count" : 0, "start_org": 0, "merge_org": 0, "declare_bankruptcy": 0, "end_org": 0}
CONFLICT = {"count" : 0, "attack": 0, "demonstrate": 0}
CONTACT = {"count" : 0, "meet": 0, "phone_write": 0}
PERSONELL = {"count" : 0, "start_position": 0, "end_position": 0, "nominate" : 0, "elect" : 0}
JUSTICE = {"count" : 0, "arrest_jail" : 0, "release_parole": 0, "trial_hearing": 0, "charge_indict" : 0, "sue" : 0, "convict" : 0, "sentence": 0, "fine": 0, "execute" : 0, "extradite" : 0, "acquit" : 0, "appeal": 0, "pardon": 0}

def statiscal(event_type, event_subtype):
  if event_type == "Life":
    LIFE["count"] += 1
    if event_subtype == "Be-Born":
      LIFE["be_born"] += 1
    elif event_subtype == "Marry":
      LIFE["marry"] += 1
    elif event_subtype == "Divorce":
      LIFE["divorce"] += 1
    elif event_subtype == "Injure":
      LIFE["injure"] += 1
    elif event_subtype == "Die":
      LIFE["die"] += 1
  elif event_type == "Movement":
    MOVEMENT["count"] += 1
    if event_subtype == "Transport":
      MOVEMENT["transport"] += 1
  elif event_type == "Transaction":
    TRANSACTION["count"] += 1
    if event_subtype == "Transfer-Ownership":
      TRANSACTION["transfer_ownership"] += 1
    elif event_subtype == "Transfer-Money":
      TRANSACTION["transfer_money"] += 1
  elif event_type == "Business":
    BUSINESS["count"] += 1
    if event_subtype == "Start-Org":
      BUSINESS["start_org"] += 1
    elif event_subtype == "Merge-Org":
      BUSINESS["merge_org"] += 1
    elif event_subtype == "Declare-Bankruptcy":
      BUSINESS["declare_bankruptcy"] += 1
    elif event_subtype == "End-Org":
      BUSINESS["end_org"] += 1
  elif event_type == "Conflict":
    CONFLICT["count"] += 1
    if event_subtype == "Attack":
      CONFLICT["attack"] += 1
    elif event_subtype == "Demonstrate":
      CONFLICT["demonstrate"] += 1
  elif event_type == "Contact":
    CONTACT["count"] += 1
    if event_subtype == "Meet":
      CONTACT["meet"] += 1
    elif event_subtype == "Phone-Write":
      CONTACT["phone_write"] += 1
  elif event_type == "Personell":
    PERSONELL["count"] += 1
    if event_subtype == "Start-Position":
      PERSONELL["start_position"] += 1
    elif event_subtype == "End-Position":
      PERSONELL["end_position"] += 1
    elif event_subtype == "Nominate":
      PERSONELL["nominate"] += 1
    elif event_subtype == "Elect":
      PERSONELL["elect"] += 1
  elif event_type == "Justice":
    JUSTICE["count"] += 1
    if event_subtype == "Arrest-Jail":
      JUSTICE["arrest_jail"] += 1
    elif event_subtype == "Release-Parole":
      JUSTICE["release_parole"] += 1
    elif event_subtype == "Trial-Hearing":
      JUSTICE["trial_hearing"] += 1
    elif event_subtype == "Charge-Indict":
      JUSTICE["charge_indict"] += 1
    elif event_subtype == "Sue":
      JUSTICE["sue"] += 1
    elif event_subtype == "Convict":
      JUSTICE["convict"] += 1
    elif event_subtype == "Sentence":
      JUSTICE["sentence"] += 1
    elif event_subtype == "Fine":
      JUSTICE["fine"] += 1
    elif event_subtype == "Execute":
      JUSTICE["execute"] += 1
    elif event_subtype == "Extradite":
      JUSTICE["extradite"] += 1
    elif event_subtype == "Acquit":
      JUSTICE["acquit"] += 1
    elif event_subtype == "Appeal":
      JUSTICE["appeal"] += 1
    elif event_subtype == "Pardon":
      JUSTICE["pardon"] += 1

def white_to_file(file_name):
  fo = open(file_name,'w')
  fo.write("LIFE : %d\n" %(LIFE["count"]))
  fo.write("  Be-Born : %d\n" %(LIFE["be_born"]))
  fo.write("  Marry : %d\n" %(LIFE["marry"]))
  fo.write("  Divorce : %d\n" %(LIFE["divorce"]))
  fo.write("  Injure : %d\n" %(LIFE["injure"]))
  fo.write("  Die : %d\n" %(LIFE["die"]))
  fo.write("MOVEMENT : %d\n" %(MOVEMENT["count"]))
  fo.write("  Transport : %d\n" %(MOVEMENT["transport"]))
  fo.write("TRANSACTION : %d\n" %(TRANSACTION["count"]))
  fo.write("  Transfer-Ownership : %d\n" %(TRANSACTION["transfer_ownership"]))
  fo.write("  Transfer-Money : %d\n" %(TRANSACTION["transfer_money"]))
  fo.write("BUSINESS : %d\n" %(BUSINESS["count"]))
  fo.write("  Start-Org : %d\n" %(BUSINESS["start_org"]))
  fo.write("  Merge-Org : %d\n" %(BUSINESS["merge_org"]))
  fo.write("  Declare-Bankruptcy : %d\n" %(BUSINESS["declare_bankruptcy"]))
  fo.write("  End-Org : %d\n" %(BUSINESS["end_org"]))
  fo.write("CONFLICT : %d\n" %(CONFLICT["count"]))
  fo.write("  Attack : %d\n" %(CONFLICT["attack"]))
  fo.write("  Demonstrate : %d\n" %(CONFLICT["demonstrate"]))
  fo.write("CONTACT : %d\n" %(CONTACT["count"]))
  fo.write("  Meet : %d\n" %(CONTACT["meet"]))
  fo.write("  Phone-Write : %d\n" %(CONTACT["phone_write"]))
  fo.write("PERSONELL : %d\n" %(PERSONELL["count"]))
  fo.write("  Start-Position : %d\n" %(PERSONELL["start_position"]))
  fo.write("  End-Position : %d\n" %(PERSONELL["end_position"]))
  fo.write("  Nominate : %d\n" %(PERSONELL["nominate"]))
  fo.write("  Elect : %d\n" %(PERSONELL["elect"]))
  fo.write("JUSTICE : %d\n" %(JUSTICE["count"]))
  fo.write("  Arrest-Jail : %d\n" %(JUSTICE["arrest_jail"]))
  fo.write("  Release-Parole : %d\n" %(JUSTICE["release_parole"]))
  fo.write("  Trial-Hearing : %d\n" %(JUSTICE["trial_hearing"]))
  fo.write("  Charge-Indict : %d\n" %(JUSTICE["charge_indict"]))
  fo.write("  Sue : %d\n" %(JUSTICE["sue"]))
  fo.write("  Convict : %d\n" %(JUSTICE["convict"]))
  fo.write("  Sentence : %d\n" %(JUSTICE["sentence"]))
  fo.write("  Fine : %d\n" %(JUSTICE["fine"]))
  fo.write("  Execute : %d\n" %(JUSTICE["execute"]))
  fo.write("  Extradite : %d\n" %(JUSTICE["extradite"]))
  fo.write("  Acquit : %d\n" %(JUSTICE["acquit"]))
  fo.write("  Appeal : %d\n" %(JUSTICE["appeal"]))
  fo.write("  Pardon : %d\n" %(JUSTICE["pardon"]))
  fo.close()

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
      del file_path[len(file_path) - 1]
      del file_path[len(file_path) - 1]
      f_list.close()
    return file_path

def get_file_path_inside(path):
  file_path = []
  data_path = '../../ace_2005_td_v7/data/English' + '/' + path
  f_list = open(data_path+'/FileList','r')
  line = f_list.readline()
  line = f_list.readline()
  while line:
    arr = line.split('\t')
    file_name = arr[0] + ".apf.xml"
    file_path.append(data_path+'/adj/'+file_name)
    line = f_list.readline()
  del file_path[len(file_path) - 1]
  del file_path[len(file_path) - 1]
  f_list.close()
  return file_path

def read_data(file_path, file_name):
  for file in file_path:
    try:
      tree = ET.parse(file)
      root = tree.getroot()
      for event in root.iter('event'):
        statiscal(event.attrib['TYPE'], event.attrib['SUBTYPE'])
        white_to_file(file_name)
    except IOError:
      pass

read_data(get_file_path(), "stastical_all.txt")
read_data(get_file_path_inside("bc"), "statiscal_bc.txt")
read_data(get_file_path_inside("bn"), "statiscal_bn.txt")
read_data(get_file_path_inside("cts"), "statiscal_cts.txt")
read_data(get_file_path_inside("nw"), "statiscal_nw.txt")
read_data(get_file_path_inside("un"), "statiscal_un.txt")
read_data(get_file_path_inside("wl"), "statiscal_wl.txt")

