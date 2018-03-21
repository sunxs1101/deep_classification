# _*_coding:utf-8_*_
import re
other_state = 0
english_state = 1
digit_state = 2
#digit_seq_span = 2


def seg(line, output = 0):
    outline = u''
    state = other_state
    chlist = ['+', '-' ,'[', ']', ':', '*', '?', '~', '!', '@', '%', '^', '=', '(', ')', '{', '}', '"', '\\', '\'', '/',
              '.', u'？',u'：',u'，',u'。',u'“',u'”', u'！',u'（',  u'）', ' ']
    flag = 0
    if not isinstance(line, unicode):
        line = unicode(line, 'utf-8')
        flag = 1

    #print line
    for i in range(len(line)):
        if line[i] in chlist:
            outline = outline + ' ' + line[i]
            state = other_state
            continue
        elif len(re.findall('\w', line[i])) > 0:
            if state == other_state:
                outline = outline + ' ' + line[i]
                state = english_state
                continue
            else:
                outline = outline + line[i]
                state = digit_state
                continue
        elif len(re.findall('\d', line[i])) > 0:
            if state == other_state:
                outline = outline + ' ' + line[i]
                state = digit_state
                continue
            else:
                outline = outline + line[i]
                state = digit_state
            continue
        else:
            outline = outline + ' ' + line[i]
            state = other_state
    if flag == 0:
        return outline
    else:
        return outline.encode('utf-8')

def simple_seg(line,vocabulary,ignoreUnk=False):
    outline = u''
    flag = 0
    if not isinstance(line, unicode):
        line = unicode(line, 'utf-8')
        flag = 1
    #print line
    for i in range(len(line)):
        if ignoreUnk and not vocabulary.has_key(line[i].encode('utf8')):
          continue
        outline = outline + ' ' + line[i]
    if flag == 0:
        return outline
    else:
        return outline.encode('utf-8')

def simple_seg_batch(line_list,vocabulary,ignoreUnk=False):
  result_list=[]
  for sentence in line_list:
    result_list.append(simple_seg(sentence,vocabulary,ignoreUnk))
  return result_list

if __name__ == '__main__':
    lines = ["imax很炫的"]
    vocab={'很':1}
    for line in lines:
       print simple_seg(line,vocab,ignoreUnk=True)

