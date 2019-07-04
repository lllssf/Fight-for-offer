# -*- coding:utf-8 -*-
# 生成动词词向量并聚类
from bert_serving.client import BertClient
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import collections

bc = BertClient()

# 提取动词库
verbs = []
with open('VERBS', 'r') as f :
    for line in f:
        line = line.strip()
        verbs.append(line)

verb_we = []
for v in verbs:
    i = 0
    verb_we.append(bc.encode([v]))

verb_we = np.array(verb_we)
n = verb_we.shape[0]
verb_vec = np.reshape(verb_we,(n,768))

estimator = KMeans(n_clusters=300)  # 构造聚类器
estimator.fit(verb_vec)  # 聚类
label_pred = estimator.labels_  # 获取聚类标签
print(label_pred)

# 建立label-序号对应的字典
label_dic = dict()
for i in range(n):
    if label_pred[i] not in label_dic:
        l = []
        l.append(i)
        label_dic[label_pred[i]] = l
    else:
        label_dic[label_pred[i]].append(i)

labels = sorted(label_dic.items(), key=lambda x:x[0])
txt = []
for i in labels:
    txt.append('第'+str(i[0]+1)+'类：')
    text = []
    for j in i[1]:
       text.append(verbs[j])
    txt.append(text)
with open('label_v', 'w') as f1:
    for t in txt:
        f1.write(str(t)+'\n')

