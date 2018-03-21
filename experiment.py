#_._coding:utf-8_._
from __future__ import division 
import re
import numpy
import tensorflow as tf
import train 
import text_cnn_decoder as td
import sys
import data_helpers
import os

source_file = './data/to_evaluate/classify_from_cp.txt'
FLAGS=tf.flags.FLAGS

def experiment():
  embedding_dim_list=[128]
  num_filters_list=[128]
  filter_sizes_list=['2,3,4']

  """  
  embedding_dim_list=[128,64]
  num_filters_list=[128,64]
  filter_sizes_list=['2,3,4','3,4,5']
  FLAGS.data_path='data/data_tmp'
  FLAGS.checkpoint_every=10
  FLAGS.evaluate_every=10
  FLAGS.num_epochs=1
  """

  FLAGS.tmp_dir=True
  recall={}
  data_set=data_helpers.DataSet(FLAGS.data_path)
  for run_time in xrange(4):
    for embedding_dim in embedding_dim_list:
      for num_filters in num_filters_list:
        for filter_sizes in filter_sizes_list:
          FLAGS.embedding_dim=embedding_dim
          FLAGS.num_filters=num_filters
          FLAGS.filter_sizes=filter_sizes
          train.train(data_set)
          model_dir=train.train_info_dict['model_dir']
          print(model_dir)
          decoder=td.TextCnnDecoder(model_dir)
          classify(source_file,os.path.join(model_dir,'classify_result'),decoder)
          ananlyzer=ResultAnanlyzer(decoder.label_inv)
          ananlyzer.ananlyze_file(os.path.join(model_dir,'classify_result'),os.path.join(model_dir,'ananlyze_result'))
          model_key_info = model_dir+"\t"+str(FLAGS.num_filters)+"\t"+str(FLAGS.embedding_dim)+"\t"+FLAGS.filter_sizes
          recall[model_key_info]=ananlyzer.result_by_domain.get('total')
  recall_sorted = sorted(recall.iteritems(),key = lambda d:d[1][4],reverse=True)
  with open('runs/experiment_result','w') as rfile:
    for one in recall_sorted:
      rfile.write(one[0]+"\t"+'\t'.join(map(str,one[1]))+'\n')  

def topn(class_dict, n = 3,sort_by_value=True):
  if sort_by_value:
    local = sorted(class_dict.iteritems(), key = lambda d:d[1], reverse = True)
  else:
    local = sorted(class_dict.iteritems(), key = lambda d:d[0], reverse = True)
  return local[:n]

def classify(in_file,out_file,decoder,n_top=3,sort_by_value=True, prob=True): 
  """
  classify recodes in in_file write classify result to out_file
  return ananlyzer instance which include statistic result
  """
  count = 0
  query_list = []
  sec_list = []
  with open(in_file,'r') as infile:
    with open(out_file,'w') as outfile:
      for line in infile:
        s = line.replace('\n', '').replace('。','').split('\t')
        sec_list.append(s[1])
        query_list.append(s[0])
        count += 1
        if count % 1000==0:
          label_result=decoder.decode_batch(query_list)
          one_batch_result=trans_one_batch(query_list,sec_list,label_result,n_top,sort_by_value,prob)
          for item in one_batch_result:
            outfile.write('\t'.join(map(str,item))+'\n')
          query_list = []
          sec_list = []
      if len(query_list)>0:
        label_result = decoder.decode_batch(query_list)
        one_batch_result=trans_one_batch(query_list,sec_list,label_result,n_top,sort_by_value,prob)
        for item in one_batch_result:
          outfile.write('\t'.join(map(str,item))+'\n')

def trans_one_batch(query,label,label_result,n_top,sort_by_value, prob):
  one_batch_result = []
  for index,item in enumerate(label_result):
    if prob:
      temp = topn(item[2],n_top,sort_by_value)
    else:
      temp = topn(item[3],n_top,sort_by_value)
    single_result=[query[index],label[index]]
    single_result.extend([x[i] for x in temp for i in xrange(2)])
    one_batch_result.append(single_result)
  return one_batch_result
  


class ResultAnanlyzer():
  """对分类结果按照类别统计准确率和召回率"""
  def __init__(self,label_inv):  
    result_by_domain ={} 
    for item in label_inv.values():
        result_by_domain[item] = [0,0,0,0]
    result_by_domain['total']=[0,0,0,0]
    self.result_by_domain=result_by_domain

  def ananlyze(self,one_result):
    result_by_domain=self.result_by_domain
    result_by_domain['total'][0]=result_by_domain.get('total',[0,0,0,0])[0]+1
    result_by_domain['total'][1]=result_by_domain.get('total',[0,0,0,0])[1]+1
    result_by_domain[one_result[2]]=result_by_domain.get(one_result[2],[0,0,0,0])
    result_by_domain[one_result[1]]=result_by_domain.get(one_result[1],[0,0,0,0])
    result_by_domain[one_result[2]]=result_by_domain.get(one_result[2],[0,0,0,0])
    result_by_domain[one_result[1]][0]=result_by_domain.get(one_result[1],[0,0,0,0])[0]+1
    result_by_domain[one_result[2]][1]=result_by_domain.get(one_result[2],[0,0,0,0])[1]+1
    if one_result[1]==one_result[2]:
      result_by_domain[one_result[1]][2]=result_by_domain.get(one_result[1],[0,0,0,0])[2]+1
      result_by_domain[one_result[1]][3]=result_by_domain.get(one_result[1],[0,0,0,0])[3]+1
      result_by_domain['total'][2]=result_by_domain.get('total',[0,0,0,0])[2]+1
      result_by_domain['total'][3]=result_by_domain.get('total',[0,0,0,0])[3]+1
    elif one_result[1]==one_result[4] or one_result[1] == one_result[6]:
      result_by_domain[one_result[1]][3]=result_by_domain.get(one_result[1],[0,0,0,0])[3]+1
      result_by_domain['total'][3]=result_by_domain.get('total',[0,0,0,0])[3]+1
      
  def compute(self):
    for domain,domain_count in self.result_by_domain.items():
      domain_count.extend(['N']*4)
      if domain_count[0]> 0:
        domain_count[4]=(domain_count[2]/domain_count[0])
        domain_count[6]=(domain_count[3]/domain_count[0])
      if domain_count[1]> 0:
        domain_count[5]=(domain_count[2]/domain_count[1])
        domain_count[7]=(domain_count[3]/domain_count[1])
      self.result_by_domain[domain]=domain_count
  
  def write_result(self,path):
    with open(path,'w') as ofile:
      for domain,domain_count in self.result_by_domain.items():
        ofile.write(domain+"\t"+'\t'.join(map(str,domain_count))+'\n')

  def ananlyze_file(self,in_file,out_file):
    with open(in_file,'r') as infile:
      for line in infile:
        item=line.strip('\n').split('\t')
        self.ananlyze(item)
    self.compute()
    self.write_result(out_file)

if __name__=="__main__":
    #experiment()  
    #FLAGS.filter_sizes='2,3,4'
    #FLAGS.num_filters=256
    #FLAGS.embedding_dim=256
    #classify(source_file,'/tmp/tmp_classify',td.TextCnnDecoder('runs/tmp/1462348291/model_store/'),40,False,False)
    
    model_dir="./runs/1475948653/model_store"
    decoder=td.TextCnnDecoder(model_dir)
    classify(source_file,os.path.join(model_dir,'classify_result'),decoder)
    ananlyzer=ResultAnanlyzer(decoder.label_inv)
    ananlyzer.ananlyze_file(os.path.join(model_dir,'classify_result'),'/tmp/tmp_ananlyze_result')
    

