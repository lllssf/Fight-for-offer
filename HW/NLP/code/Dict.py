# -*- coding: utf-8 -*-
# 构建词典
import os,re
from pyltp import Segmentor, Postagger

# 模型路径
MODELDIR=os.path.join("ltp_data")

txt = []
with open('data','r') as f :
    for line in f:
        line = line.strip()
        txt.append(line)

STEP = []
for line in txt:
    # 分词
    segmentor = Segmentor()
    segmentor.load_with_lexicon(os.path.join(MODELDIR, "cws.model"), 'Dic.txt')
    words = list(segmentor.segment(line))
    # 词性标注
    postagger = Postagger()
    postagger.load(os.path.join(MODELDIR, "pos.model"))
    postags = postagger.postag(words)
    for (i,tag) in enumerate(postags):
        if tag == 'v' and words[i] not in STEP:
            print(words[i])
            STEP.append(words[i])

    # 释放模型
    segmentor.release()
    postagger.release()


with open('dict', 'w') as f1:
    for s in STEP:
        f1.write(s + '  v' + '\n')
