def sentence_seg(para):
    sentence = []
    quote  = re.split(r'(\w*：?“.*?”\D*\W?|\w*:?".*?"\D*\W?|\w*【.*?】\D*\W?|{.*?}\D*\W?)',para,re.M) # 将带引用符号的子句拆分，本数据集中只有这四种引用符

    for q in quote:
        f = re.search(r'[‘|“|【|\[|\(|\'|\"]',q) # 对不带引用符号的句子进一步拆分
        if  f == None:
            q = re.sub('([，。；！？])(.*?)', r"\1\n\2",q,re.M) # 按中文标点拆分
            q = re.sub('(.*)(\d\.\.*)',r"\1\n\2",q,re.M) # 将形如“1.balabala”的句子拆开
            q = q.split('\n')
            for s in q:
                if s != '':
                    sentence.append(s)
        else:
            sentence.append(q)
    return sentence

