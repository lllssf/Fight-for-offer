# -*- coding:utf-8 -*-
from bert_serving.client import BertClient
import numpy as np
bc = BertClient()

# 计算cos相似度
def cosine(a,b):
    return a.dot(b)/(np.linalg.norm(a)*np.linalg.norm(b))


emb=np.array(bc.encode(['登录', '登入']))
print(['登录', '登入'],":",cosine(emb[0],emb[1]))
