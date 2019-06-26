# NLP 学习
## 入门建议
1. 了解NLP最基本的知识：《Speech and Language Processing》
2. 了解早年经典的NLP模型以及论文：例如机器翻译中的IBM模型
3. 了解机器学习的基本模型：吴恩达的 machine learning 
4. 多看NLP其他子领域的论文：NLP 有很多子领域，MT，信息抽取，parsing，tagging，情感分析，MRC 等等
5. 了解 CV 和 data mining 领域的基本重大进展
6. 推荐书目：
![booklist](https://pic1.zhimg.com/v2-07af312990ad8cdf850c4b94eeecbfc8_r.jpg)
## NLP分析技术
1. 词法分析（lexical analysis）：
    - 分词（word segmentation/tokenization）
    - 词性标注（part-of-speech tag）
    - 命名实体识别（named entity recognize）
2. 句法分析（syntatic parsing）：
    - 短句结构句法分析（constituent parsing）
    - 依存句法分析（dependency syntactic parsing）
    - 深层文法句法分析
3. 语义分析（semantic parsing）：
    - 词汇级语义分析：
        - 词义消歧（word sense disambiguation）
        - 词义表示和学习（word representation）
    - 句子级语义分析：
        - 语义角色标注（semantic role labeling，SRL）
        - 深层语义分析：语义依存分析（semantic dependency parsing）
    - 篇章级语义分析：
        - 篇章连接词识别
        - 论元识别
        - 显式篇章关系识别
        - 隐式篇章关系识别

![avatar](https://pic3.zhimg.com/v2-3d2cc9e84d5912dac812dc51ddee54fa_r.jpg)
## 正则表达式

## 预训练
### word2vec - Google
```> pip install --upgrade gensim```
1. 词向量： One-hot representation --> Distributed Representation --> 计算欧氏距离或cos距离得到词语语义距离
2. 语言模型： N-gram, N-pos(part of speech) --> 拟合P
3. Word2vec有两种类型，每种类型有两种策略
4. CBOW加层次的网络结构：输入层（词向量）-->隐藏层（累加和）-->输出层（霍夫曼树）——判断一句话是否是自然语言
5. 参考资料：https://blog.csdn.net/Mr_tyting/article/details/80091842

### BERT - Google
Bert_base & Bert_large
1. 参考资料：
    - https://zhuanlan.zhihu.com/p/49271699
    - https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650751075&idx=2&sn=0a3ecd1af5f8549051760775e34db342&chksm=871a841db06d0d0bcf3cc4e620bb384e050ba6e92224d338a8ddc1543add97a4a4e7919ebf15&scene=21#wechat_redirect
    - https://www.jiqizhixin.com/articles/2019-02-18-12?from=synced&keyword=NLP
2. 输入是一个线性序列，支持单句文本和句对文本，句首用符号[CLS]表示，句尾用符号[SEP]表示，如果是句对，句子之间添加符号[SEP]。输入特征，由**Token向量、Segment向量和Position向量**三个共同组成，分别代表单词信息、句子信息、位置信息。
3. MLM随机地掩盖15%的单词，然后对掩盖的单词做预测任务，此类处理的缺点是：
    - 预训练阶段随机用符号[MASK]替换掩盖的单词，而下游任务微调阶段并没有Mask操作，会造成预训练跟微调阶段的不匹配
    - 预训练阶段只对15%被掩盖的单词进行预测，而不是整个句子，模型收敛需要花更多时间
    为解决第一个缺点，随机掩盖的单词80%用符号[MASK]替换，10%用其他单词替换，10%不做替换操作。
4. NSP，预测下一句模型，增加对句子A和B关系的预测任务，50%的时间里B是A的下一句，分类标签为IsNext，另外50%的时间里B是随机挑选的句子，并不是A的下一句，分类标签为NotNext。

### 训练中文词向量
- Word2vec https://blog.csdn.net/lilong117194/article/details/82849054
- BERT 
        - https://blog.csdn.net/zhylhy520/article/details/87615772
        - https://blog.csdn.net/qq_29660957/article/details/88683823
## 中文分词开源工具： 
- python：
    - jieba: https://github.com/fxsjy/jieba
    - SnowNLP: https://github.com/isnowfy/snownlp
    - PkuSeg: https://github.com/lancopku/pkuseg-python
    - THULAC: https://github.com/thunlp/THULAC-Python
    - HanLP: https://github.com/hankcs/pyhanlp
    - FoolNLTK: https://github.com/rockyzhengwu/FoolNLTK
    - pyltp: https://github.com/HIT-SCIR/pyltp
- Java： Paoding（准确率、分词速度、新词识别等，最棒）、mmseg4j（切分速度、准确率较高）、IKAnalyzer、Imdict-chinese-analyzer、Ansj、盘古分词
- 当前新论文：
    - Toward Fast and Accurate Neural Chinese Word Segmentation with Multi-Criteria Learning https://arxiv.org/pdf/1903.04190.pdf
    - State-of-the-art Chinese Word Segmentation with Bi-LSTMs https://aclweb.org/anthology/D18-1529

## NLP工具：
- NLTK（Natural Language Toolkit）
    1. 官方教程：http://www.nltk.org/book/
    2. 在NLTK中使用Stanford中文工具包教程：
        - http://www.zmonster.me/2016/06/08/use-stanford-nlp-package-in-nltk.html
        - https://www.cnblogs.com/baiboy/p/nltk1.html
        - http://www.52nlp.cn/python%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86%E5%AE%9E%E8%B7%B5-%E5%9C%A8nltk%E4%B8%AD%E4%BD%BF%E7%94%A8%E6%96%AF%E5%9D%A6%E7%A6%8F%E4%B8%AD%E6%96%87%E5%88%86%E8%AF%8D%E5%99%A8
- FastNLP
https://github.com/fastnlp/fastNLP
- 哈工大语言云**LTP**: Language Technology Platform
https://github.com/HIT-SCIR/ltp | http://www.ltp-cloud.com/
    - python接口：pyltp 安装
    下载pyltp：http://mlln.cn/2018/01/31/pyltp%E5%9C%A8windows%E4%B8%8B%E7%9A%84%E7%BC%96%E8%AF%91%E5%AE%89%E8%A3%85/pyltp-0.2.1-cp36-cp36m-win_amd64.whl
    ```
    > pip install path\pyltp-0.2.1-cp36-cp36m-win_amd64.whl
    ```
    - 说明文档：https://pyltp.readthedocs.io/zh_CN/latest/api.html#id19
## 书籍：Speech and Language Processing(3rd)
https://web.stanford.edu/~jurafsky/slp3/
## 网站：NLP-progress - 最新研究进展和模型
http://nlpprogre    ss.com/
## 论文

## 数据：
1. 腾讯AI Lab开源中文词向量数据：https://ai.tencent.com/ailab/nlp/embedding.html
2. 中文维基百科数据： https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2
## 项目实践
NLP入门实例推荐（Tensorflow实现）https://blog.csdn.net/Irving_zhang/article/details/69396923
