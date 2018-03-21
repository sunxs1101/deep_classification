#-.-coding: utf-8 -.-
import numpy as np
import tensorflow as tf
from text_cnn import TextCNN
import data_helpers
import sys
import time
import timeit
import data_helper
from tensorflow.python.platform import gfile
import segmenter
import os
from tensorflow.python.tools import inspect_checkpoint


# Training parameters
"""
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
"""
# Misc Parameters
"""
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_string("model_path", "./runs/checkpoints_server/best_checkpoints", "model path ")
#tf.flags.DEFINE_string("data_path", "./data/data_server", "数据存放文件夹")
tf.flags.DEFINE_string("label_inv_path", "./runs/label_inv_server", "dir where label_inv are")
"""

FLAGS = tf.flags.FLAGS
"""
session_conf = tf.ConfigProto(
  allow_soft_placement=FLAGS.allow_soft_placement,
  log_device_placement=FLAGS.log_device_placement,
  intra_op_parallelism_threads=20)
sess = tf.Session(config=session_conf)
sess.run(tf.initialize_all_variables())
"""
"""
with sess.as_default():
  cnn = TextCNN(
      sequence_length=20,
      num_classes=len(label_inv),
      vocab_size=len(vocabulary),
      embedding_size=FLAGS.embedding_dim,
      filter_sizes=map(int, FLAGS.filter_sizes.split(",")),
      num_filters=FLAGS.num_filters,
      l2_reg_lambda=FLAGS.l2_reg_lambda)
"""
class TextCnnDecoder():
  def __init__(self,model_store_path):
    label_inv={}
    vocabulary={}
    label_inv_path=os.path.join(model_store_path,'model_store/label_inv')
    with open(label_inv_path,'r') as f:
      for line in f:
        split=line.strip('\n').split(' ')
        label_inv[int(split[0])]=split[1]
    vocab_path=os.path.join(model_store_path,'model_store/vocabulary')
    with open(vocab_path,'r') as f:
      for line in f:
        split=line.strip('\n').split(' ')
        vocabulary[split[0]]=int(split[1])
    model_info_path = os.path.join(model_store_path, 'model_store/model_info')
    with open(model_info_path, 'r') as f:
      model_info_dict={}
      for line in f:
        split=line.strip('\n').split('=')
        model_info_dict[split[0]] = split[1]
      if model_info_dict['embedding_dim']:
        FLAGS.embeding_dim = int(model_info_dict['embedding_dim'])
      if model_info_dict['filter_sizes']:
        FLAGS.filter_sizes = model_info_dict['filter_sizes']
      if model_info_dict['num_filters']:
        FLAGS.num_filters = int(model_info_dict['num_filters'])
    print(model_info_dict)
    """restore model from checkpoint"""
    with tf.Graph().as_default():
      session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement,
        intra_op_parallelism_threads=20)
      sess = tf.Session(config=session_conf)
      sess.run(tf.initialize_all_variables())
      with sess.as_default():
        cnn = TextCNN(
            sequence_length=20,
            num_classes=len(label_inv),
            vocab_size=len(vocabulary),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=map(int, FLAGS.filter_sizes.split(",")),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)
        model_checkpoint_path=os.path.join(model_store_path,'checkpoints/model-159000')
        if gfile.Exists(model_checkpoint_path+'.index'): #inspect_checkpoint.print_tensors_in_checkpoint_file(model_checkpoint_path,"",False):#
          print("Reading model parameters from %s" % model_checkpoint_path)
          saver = tf.train.Saver(tf.all_variables())
          saver.restore(sess,model_checkpoint_path)
        else:
          print("file %s not exist." % model_checkpoint_path)
          exit(1)
          sess.run(tf.initialize_all_variables())
        print("Model restored.")
    self.cnn=cnn
    self.sess=sess
    self.label_inv=label_inv
    self.vocabulary=vocabulary

  def decode(self,query):
      """
      sentence=' '.join(list(unicode(sentence,'utf-8')))
      sentence_padded = data_helper.pad_origin_sentence(sentence,'%p%',20)
      id_sentence = np.array([[vocabulary[word.encode('utf-8')] for word in sentence] for sentence in sentence_padded])
      """
      cnn=self.cnn
      vocabulary=self.vocabulary
      label_inv=self.label_inv
      sess=self.sess
      filtered_sentence=segmenter.simple_seg(query,vocabulary,ignoreUnk=True)
      sentence_padded = data_helper.pad_origin_sentence(filtered_sentence,'%p%',20)
      id_sentence = np.array([[vocabulary[word] for word in sentence] for sentence in sentence_padded])
      feed_dict = {
          self.cnn.input_x: id_sentence,
          self.cnn.dropout_keep_prob:1.0 
      }
      predictions, scores,probs = sess.run([cnn.predictions, cnn.scores,cnn.probs],feed_dict)
      #prob=tf.Tensor.eval(tf.nn.softmax(scores),session=tf.Session())
      #prob=tf.Tensor.eval(tf.nn.softmax(scores),session=sess)
      #prob=tf.nn.softmax(scores).eval()
      #prob=tf.nn.softmax(scores)
      score_map={}
      prob_map={}
      for i in range(len(scores[0])):
        score_map[label_inv[i]]=scores[0][i]
        prob_map[label_inv[i]]=probs[0][i]
      return label_inv[predictions[0]] #,np.amax(probs),prob_map,score_map
  
  def decode_batch(self,query_list):
      vocabulary=self.vocabulary
      label_inv=self.label_inv
      cnn=self.cnn
      sess=self.sess

      filtered_sentence=segmenter.simple_seg_batch(query_list,vocabulary,ignoreUnk=True)
      sentence_padded = data_helper.pad_origin_sentence_batch(filtered_sentence,'%p%',20)
      id_sentence = np.array([[vocabulary[word] for word in sentence] for sentence in sentence_padded])
      feed_dict = {
          cnn.input_x: id_sentence,
          cnn.dropout_keep_prob: 1.0
      }
      predictions, scores ,prob= sess.run([cnn.predictions, cnn.scores,cnn.probs],feed_dict)
      result_list=[]
      for index,one_predict in enumerate(predictions):
        score_map={}
        prob_map={}
        for i in range(len(scores[index])):
          score_map[label_inv[i]]=scores[index][i]
          prob_map[label_inv[i]]=prob[index][i]
        result_list.append([label_inv[one_predict],np.amax(prob[index]),prob_map,score_map])
      return result_list

  def decode_from_console(self):
    for i in os.listdir("data/testData/"):
      print i
      fullname = os.path.join("data/testData/", i)
      label = i.split('_')[0]
      n = 0 
      m = 0
      with open(fullname,'r') as f:
        for line in f:
          m = m + 1
          result = self.decode(line.split('\t')[1].strip())
          if result == label:
            n = n + 1
      print fullname,label, n, m, float(n)/m

'''
    sentence=sys.stdin.readline()
    while sentence:
      start=time.time()
      #print(timeit.timeit(stmt='decode(sentence)',setup='from __main__ import decode;sentence="你好"'))
      result=self.decode(sentence)
      print time.time()-start
      print result
      sentence=sys.stdin.readline()
'''

  

#decode_train_data('data/train_decode.txt')

if __name__=='__main__':
  decoder=TextCnnDecoder('runs/1512096928/')
  decoder.decode_from_console()
