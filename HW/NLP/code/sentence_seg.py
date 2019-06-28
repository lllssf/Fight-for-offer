# 分句: ‘’“”【】[]''中的内容不能分割
text = '1.玄德幼时，与乡中小儿戏于树下，曰：“我为天子，当乘此车盖。”2.行吧，【唉呀妈呀，好的吧】'

def sentence_seg(para):
    sentence = []
    quote  = re.sub(r'(.*[，。！ ])(.*[‘|“|【|\[|\(|\'].*[’|”|】|\]|\'])', r'\1\n\2',para,re.M)
    quote = quote.split()
    for q in quote:
        if "“" not in q:
            q = re.sub('([，！？\?  ])([^”’])', r"\1\n\2",q,re.M)
            q = re.sub('(.*)(\d.*)',r"\1\n\2",q,re.M)
            q = q.split('\n')
            for s in q:
                sentence.append(s)
        else:
            sentence.append(q)
    return sentence


s = sentence_seg(text)
print('\n'.join(s))
