# -*- coding: utf-8 -*-
import  re
txt = []
with open('data.txt','r') as f :
    for line in  f:
        line = line.strip()
        txt.append(re.sub('<[^>]*>', '', line)) #去除<>及其中内容

with open('clean.txt','w') as f1:
    for s in txt:
        f1.write(s)
        f1.write('\n')
