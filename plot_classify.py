from time import time

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, decomposition, ensemble,
                discriminant_analysis, random_projection)

def plot_classify_result(file_path,min_c,max_c,relative):
  file_list=os.listdir(file_path);
  os.chdir(file_path) 
  i=0
  for one_file in file_list:
    if os.path.isfile(one_file):
      file_name=os.path.basename(one_file)
      probs=[]
      with open(one_file,'r') as f:
        for one_line in f:
           probs.append(float(one_line.split('\t')[2]))
      if(min_c<=len(probs) and len(probs)<max_c): 
        probs.append(1.01)
        if relative:
          x=[float(i)/len(probs) for i in xrange(len(probs))]
          plt.plot(x,probs,label=file_name)
        else:
          plt.plot(probs,label=file_name)
  plt.legend(loc=4)
  plt.savefig(os.path.join("/tmp","classify_result_{}_{}_{}".format(min_c,max_c,relative)))
  plt.close()
  plt.show()


def plot_recall():
  plot_classify_result('/home/chong/share/split3',100000,1000000,True)
  plot_classify_result('/home/chong/share/split3',50000,100000,True)
  plot_classify_result('/home/chong/share/split3',30000,50000,True)
  plot_classify_result('/home/chong/share/split3',10000,30000,True)
  plot_classify_result('/home/chong/share/split3',1000,10000,True)
  plot_classify_result('/home/chong/share/split3',1,1000,True)

n_features = 38

X = []
y = []

def read_classify_result(classify_result_file):
  with open(classify_result_file,'r') as in_file:
    count=0;
    for line in in_file:
      split_line = line.strip('\n').split('\t');
      if len(split_line) != 78:
        print(str(len(split_line))+"\t"+line)
        continue
      label = split_line[1]
      predict=[split_line[i*2] for i in xrange(1,39)]
      if label not in predict:
        continue
      scores=[float(split_line[i*2+1]) for i in xrange(1,39)]
      X.append(scores)
      y.append(predict.index(label))
      if count>2000:
        return
      count=count+1
# Scale and visualize the embedding vectors
def plot_embedding(X, title=None):
    x_min, x_max = np.min(X,0), np.max(X,0)
    X = (X-x_min) / (x_max - x_min)

    plt.figure()
    for i in range(X.shape[0]):
        plt.text(X[i,0], X[i,1], str(y[i]),
            color = plt.cm.Set1(y[i]/40.),
            fontdict = {'weight':'bold', 'size':9})
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

def plot_pca():
  ### Projection on to the first 2 principal components
  print "Computing PCA projection"
  t0 = time()
  X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(X)
  plot_embedding(X_pca, "Principal Components projection (time %.2fs)" %(time() - t0))
  plt.show()

def plot_tsne():  
  ### t-SNE embedding
  print "Computing t-SNE embedding"
  tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
  t0 = time()
  print(len(X))
  X_tsne = tsne.fit_transform(X)
  plot_embedding(X_tsne, "t-SNE embedding (time %.2fs)" %(time() - t0))
  
  plt.show()




if __name__ == "__main__":
  read_classify_result('/tmp/tmp_classify')
  plot_pca() 
