#-.-coding: utf-8 -.-
#from pypinyin import pinyin,lazy_pinyin
import itertools
from collections import Counter
import numpy as np
import data_helper
def read_sentence(file_path,standard_len):
  sentence_list=[]  
  with open(file_path,'r') as f:
    for sentence in f:
       words=sentence.strip('\n').split(' ');
       sentence_list.append(pad_sentence(words,'%p%',20))
  return sentence_list

def pad_origin_sentence(sentence,pad_string,std_len):
    sentence_list=[]
    words=sentence.split(' ')
    sentence_list.append(pad_sentence(words,'%p%',20))
    return sentence_list

def pad_origin_sentence_batch(sentence_list,pad_string,std_len):
    result_list=[]
    for one in sentence_list:
      result_list.append(pad_origin_sentence(one,pad_string,std_len)[0])
    return result_list

def pad_sentence(sentence,pad_string,std_len):
  result=["^"]
  if len(sentence)>=(std_len-1):
    result.extend(sentence[0:std_len-1])
    return result
  else:
    result.extend(sentence)
    result.extend([pad_string]*(std_len-len(sentence)-1))
    return result

def read_label(label_path):
  label_list=[]
  with open(label_path,'r') as f:
    for label in f:
      label_list.append(label.strip('\n'))
  label_dict = {x: i for i, x in enumerate(set(label_list))}
  print(label_dict)
  label_one_hot=np.zeros([len(label_list),len(label_dict)])
  for i in range(len(label_list)):
    label_one_hot[i,label_dict[label_list[i]]]=1
  return label_one_hot,{value:key for key, value in label_dict.items()}
  



def build_vocab(sentences):
    """ 
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def conv2pinyin(sentence):
  return lazy_pinyin(sentence)

