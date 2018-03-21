#_q._coding:utf-8_._
from __future__ import division
import urllib2
import re
import traceback
import json
#import text_cnn_decoder as td
input_file = './data/origin_test_clean_nochat.txt'
result_file = './runs/evaluate/classify_result_nochat.txt'
classify_error_file = './runs/evaluate/classify_error_nochat.txt'
run_error_file = './runs/evaluate/inputerror_nochat.txt'
label_dict = {'appmgr':u'应用','translation':u'翻译','music':u'音乐','video':u'视频','map':u'地图','localsearch':u'周边','weather':u'天气','chat':u'聊天','setting':u'设置','stock':u'股票','tv':u'电视','setting.tv':u'TV设置','flight':u'航班','hotline':u'黄页','contact':u'联系人','websearch':u'搜索','novel':u'小说','calculator':u'计算器','note':u'备忘','traffic.control':u'限行','website':u'网站','microblog':u'微博','cookbook':u'菜谱','news':u'新闻','alarm':u'闹铃','sms':u'短信','reminder':u'提醒','call':u'电话','train':u'火车','movie':u'电影','map':u'地图','calendar':u'日历','traffic':u'路况'}
label_inv_dict=dict([(value,key) for key,value in label_dict.items()])
print(label_inv_dict)

total_count_domain ={} 
correct_count_domain = {}
classify_count_domain={}
top3_count_domain={}

for item in label_dict.keys():
    total_count_domain[item] = 0
    correct_count_domain[item] = 0
    classify_count_domain[item] = 0
    top3_count_domain[item] = 0

def per(input_file):
  total_count = 0
  total_correct = 0
  total_top3_correct = 0
  error_count = 0

  classify_error_list=[]
  run_error_list=[]
  with open(input_file, 'r') as f:
    for item in f:
        print total_count
        try:
            tomatch,toinput=item.strip('\n').split('\t')
            #print tomatch
            tomatch=unicode(tomatch,'utf-8')
            if not label_inv_dict.has_key(tomatch):
              continue
            to_match_label=label_inv_dict[tomatch]
            toinput = toinput.replace(' ', '')
            #print toinput
            #a = urllib2.urlopen("http://10.10.10.135:8091/service/classify?ntop=3&text=%s"%toinput).read()
            a = urllib2.urlopen("http://10.10.20.213:8080/magnus-web/classify?ntop=3&text=%s"%toinput).read()
            """
            label = a.split('"')[3]
            if label == 'setting.mp':
              label='setting'
            if label == 'audio':
              label='music'
            #print label
            if to_match_label == label:
                total_correct = total_correct + 1
                total_top3_correct +=  1
                correct_count_domain[label] = correct_count_domain[label] + 1
                top3_count_domain[to_match_label] +=1
            else:
                classify_error_list.append(item.strip('\n')+"\t"+label+"\t"+label_dict[label].encode('utf-8')+"\n")
                n_top=json.loads(a)['top']
                if to_match_label in n_top:
                  top3_count_domain[to_match_label] +=1
                  total_top3_correct +=  1
            classify_count_domain[label] += 1
            total_count = total_count + 1
            #print(tomatch)
            total_count_domain[label_inv_dict[tomatch]]+=1
            """
            total_count = total_count + 1
        except KeyboardInterrupt, e:
            exit("")
        except:
            traceback.print_exc()
            error_count = error_count + 1
            run_error_list.append(item)
            continue

  result_list=[]  
  
  result_list.append("total\t"+str(total_correct)+"\t"+str(total_top3_correct)+"\t"+str(total_count)+"\t"+str(total_count)+"\t"+str(total_correct/total_count)+"\t"+str(total_top3_correct/total_count)+"\t"+str(total_correct/total_count)+"\n")
  for label in label_dict:
    recall=0
    if total_count_domain[label]>0:
      recall=correct_count_domain[label]/total_count_domain[label]
    top3_recall=0
    if total_count_domain[label]>0:
      top3_recall=top3_count_domain[label]/total_count_domain[label]
    prec = 0
    if classify_count_domain[label]>0:
      prec=correct_count_domain[label]/classify_count_domain[label]
    result_list.append(label+"\t"+str(correct_count_domain[label])+"\t"+str(top3_count_domain[label])+"\t"+str(total_count_domain[label])+"\t"+str(classify_count_domain[label])+"\t"+str(recall)+"\t"+str(top3_recall)+"\t"+str(prec)+"\n")

  with open(result_file,'w') as f:
    for line in result_list:
      f.write(line)
  with open(classify_error_file,'w') as f:
    for line in classify_error_list:
      f.write(line) 

  with open(run_error_file,'w') as f:
    for line in run_error_list:
      f.write(line) 

if __name__=="__main__":
    per(input_file) 
