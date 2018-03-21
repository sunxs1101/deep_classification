#--.--coding:utf-8 --.--
import sys
import random
import os


service2cate_path='./conf/service2cate.txt'
service2cate={}
with open(service2cate_path,'r') as f:
  for line in f:
    service,label=line.strip('\n').split('\t');
    service2cate[service]=label

def preprocess(in_path,out_dir):
  files=os.listdir(in_path)
  origin_dir=os.getcwd()
  os.chdir(in_path)
  query_list=[]
  label_list=[]
  input_set=set()
  for one_file in files:
    if os.path.isfile(one_file):
      with open(one_file,'r') as f:
        for line in f:
          if line.find('\t')==-1:
            continue
          if line in input_set:
            continue
          input_set.add(line)
          query,service=line.strip('\n').split("\t")
          query=query.replace('。','').replace('？','').replace('！','')
          label=convert(service)
          query_list.append(query)
          label_list.append(label)
  assert len(query_list)==len(label_list)
  index_list=[i for i in xrange(len(query_list))]
  random.shuffle(index_list)
  query_path=os.path.join(out_dir,'train_query.txt');
  seg_path=os.path.join(out_dir,'train_query_seg.txt');
  label_path=os.path.join(out_dir,'train_label.txt');
  os.chdir(origin_dir)
  with open(query_path,'w') as f_query,open(label_path,'w') as f_label , open(seg_path,'w') as f_seg:
    for index in index_list:
      f_query.write(query_list[index]+'\n')
      f_seg.write(seg_sentence(query_list[index])+'\n')
      f_label.write(label_list[index]+'\n')


def convert(service):
   if not service2cate.has_key(service):
     print 'no responding label for service {}'.format(service)
     exit(1)
   return service2cate.get(service)
  
def seg_sentence(line):
    flag = 0
    if not isinstance(line, unicode):
        line = unicode(line, 'utf-8')
        flag = 1
    out=' '.join([x for x in line])
    out=' '.join(out.split()) #将多个空格替换为一个空格
    if flag == 0:
      return out 
    else:
      return out.encode('utf-8')

def main():
  preprocess(sys.argv[1],sys.argv[2])

if __name__=='__main__':
  main() 
