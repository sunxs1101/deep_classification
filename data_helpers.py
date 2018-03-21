import numpy as np
import re
import itertools
from collections import Counter
import data_helper
import os
import random
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

class DataSet():
  def __init__(self,data_path):
    # Load data
    print("Loading data...")
    self.x, self.y, self.vocabulary, self.vocabulary_inv, self.label_inv = load_data(data_path)
    # Randomly shuffle data
    np.random.seed(10) #
    shuffle_indices = np.random.permutation(np.arange(len(self.y)))
    x_shuffled = self.x[shuffle_indices]
    y_shuffled = self.y[shuffle_indices]
    dev_num = len(x_shuffled)/10 if len(x_shuffled)<100000 else 10000
    
    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    self.x_train, self.x_dev = x_shuffled[:-dev_num], x_shuffled[-dev_num:]
    self.y_train, self.y_dev = y_shuffled[:-dev_num], y_shuffled[-dev_num:]
    print("Vocabulary Size: {:d}".format(len(self.vocabulary)))
    print("Train/Dev split: {:d}/{:d}".format(len(self.y_train), len(self.y_dev)))

def load_data_and_labels(data_path):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    query = data_helper.read_sentence(os.path.join(data_path,'train_query_seg.txt'),20)
    labels,label_inv =data_helper.read_label(os.path.join(data_path,'train_label.txt'))
    return [query, labels,label_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]

def noise_input_data(id_sentences,labels,vocabulary,min_count=1000):
    label_count=np.sum(labels,axis=0)
    repeat_num = min_count/label_count
    noise_sentence=[]
    noise_label=[]
    for index,one_sentence in enumerate(id_sentences):
      one_repeat=repeat_num[np.where(labels[index]==1)] 
      if one_repeat >1:
         noise_sentence.extend(add_noise(one_sentence,len(vocabulary),one_repeat))
         for i in xrange(one_repeat):
           noise_label.append(labels[index])
    if len(noise_sentence) >0 and len(noise_label)>0:
      id_sentences=np.vstack((id_sentences,np.array(noise_sentence)))
      labels=np.vstack((labels,np.array(noise_label)))
    return id_sentences,labels

def add_noise(id_sentence,max_id,repeat_num):
    result=[]
    for i in xrange(repeat_num):
      if len(id_sentence)<=3:
        if random.random()<0.5:
          gen_sentence=np.append(random.randint(0,max_id-1),id_sentence)
        else:
          gen_sentence=np.append(id_sentence,random.randint(0,max_id-1))
          result.append(gen_sentence)
      else:
        gen_sentence=np.array(id_sentence)
        gen_sentence[random.randint(0,np.shape(np.where(gen_sentence>0))[1]-1)]=random.randint(0,max_id-1)
        result.append(gen_sentence)
    return result      
      



def load_data(data_path):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels,label_inv = load_data_and_labels(data_path)
    vocabulary, vocabulary_inv =data_helper.build_vocab(sentences)
    x, y = build_input_data(sentences, labels, vocabulary)
    print('before adding noise train case number in every category')
    count_category(y,label_inv)
    x,y=noise_input_data(x,y,vocabulary)
    print('after adding noise train case number in every category')
    count_category(y,label_inv)
    return [x, y, vocabulary, vocabulary_inv,label_inv]

def count_category(labels,label_inv):
    label_count=np.sum(labels,axis=0)
    print({label_inv[index]:int(value) for index,value in enumerate(label_count)})


def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

#load_data('data/data_server')
