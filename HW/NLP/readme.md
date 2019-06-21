# NLP任务攻坚筹备工作 
## 任务流程：
1. 中文分词 --> 词性标注 --> 依存句法分析
2. 方法：
    - 序列标注方法（Sequence Labeling）：给字打标签，很难利用词级别的信息
    - 基于转移的方法（Transition-based Method）：通过转移动作序列进行分词，即从左往右判断相邻的字是否分开。
    - 基于图的方法(Graph-based Method)
### 中文分词（Chinese Word Segmentation，CWS）
1. 将连续的字序列按照一定的规范切分成词序列
2. 中文分词准确率低的原因：词表收录、分词规范、新词识别、歧义切分
3. 中文分词算法：
    - 基于字符串匹配：最大匹配、最少切分、改进扫描
    - 基于统计模型的序列标注切分：N元语言模型、信道-噪声模型、HMM、CRF(Conditional random field)条件随机场
    - 综合方法
4. 中文分词开源工具： 
    - python：
        - jieba: https://github.com/fxsjy/jieba
        - SnowNLP: https://github.com/isnowfy/snownlp
        - PkuSeg: https://github.com/lancopku/pkuseg-python
        - THULAC: https://github.com/thunlp/THULAC-Python
        - HanLP: https://github.com/hankcs/pyhanlp
        - FoolNLTK: https://github.com/rockyzhengwu/FoolNLTK
        - pyltp: https://github.com/HIT-SCIR/pyltp
    - Java： Paoding（准确率、分词速度、新词识别等，最棒）、mmseg4j（切分速度、准确率较高）、IKAnalyzer、Imdict-chinese-analyzer、Ansj、盘古分词
5. 目前最准确的模型：
    - Toward Fast and Accurate Neural Chinese Word Segmentation with Multi-Criteria Learning https://arxiv.org/pdf/1903.04190.pdf
    - State-of-the-art Chinese Word Segmentation with Bi-LSTMs https://aclweb.org/anthology/D18-1529
### 词性标注（POS tagging）
词性标注一般比较简单，所以多与其他任务相结合
1. 基于字的序列标注方法：使用“BMES”和词性的交叉标签来给每个字打标签。
2. 基于转移的方法：先利用BiLSTM编码器来提取上下文特征，在解码时每一步都预测一个动作，动作的候选集和为是否分词以及词性。
### 依存句法分析（Parsing）
1. 基于转移的方法：Stack LSTM，通过三个LSTM来建模栈状态、待输入序列和动作序列
2. 基于图的方法：Biaffine模型（最流行）
### 词语相似度
1. 基于世界知识（Ontology），一般利用同义词词典等 -> 计算词语语义距离
2. 基于大规模的语料来统计， -> 词语的相关性
3. 《知网》是一个以汉语和英语的词语所代表的概念为描述对象，以揭示概念与概念之间以及概念所具有的属性之间的关系为基本内容的常识知识库。：
    - 概念：对词汇语义的描述
    - 义原：知识表示语言（描述一个”概念“的最小意义单位），《知网》共有1500义原
### 中文输入纠错

### 词法分析
难点： 新词、错别字、谐音字、非规范词、歧义
#### 歧义消除
## 入门建议
1. **了解NLP最基本的知识**：《Speech and Language Processing》
2. **了解早年经典的NLP模型以及论文**：例如机器翻译中的IBM模型
3. **了解机器学习的基本模型**：吴恩达的 machine learning 
4. **多看NLP其他子领域的论文**：NLP 有很多子领域，MT，信息抽取，parsing，tagging，情感分析，MRC 等等
5. **了解 CV 和 data mining 领域的基本重大进展**

## 工具：
- NLTK（Natural Language Toolkit）
    1. 官方教程：http://www.nltk.org/book/
    2. 在NLTK中使用Stanford中文工具包教程：
        - http://www.zmonster.me/2016/06/08/use-stanford-nlp-package-in-nltk.html
        - https://www.cnblogs.com/baiboy/p/nltk1.html
        - http://www.52nlp.cn/python%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86%E5%AE%9E%E8%B7%B5-%E5%9C%A8nltk%E4%B8%AD%E4%BD%BF%E7%94%A8%E6%96%AF%E5%9D%A6%E7%A6%8F%E4%B8%AD%E6%96%87%E5%88%86%E8%AF%8D%E5%99%A8
- FastNLP
https://github.com/fastnlp/fastNLP
- LTP: Language Technology Platform
https://github.com/HIT-SCIR/ltp
## 书籍：Speech and Language Processing(3rd)
https://web.stanford.edu/~jurafsky/slp3/
## 网站：NLP-progress - 最新研究进展和模型
http://nlpprogress.com/
## word2vec - Google
**pip install --upgrade gensim**
1. 词向量： One-hot representation --> Distributed Representation --> 计算欧氏距离或cos距离得到词语语义距离
2. 语言模型： N-gram, N-pos(part of speech) --> 拟合P
3. Word2vec有两种类型，每种类型有两种策略
4. CBOW加层次的网络结构：输入层（词向量）-->隐藏层（累加和）-->输出层（霍夫曼树）——判断一句话是否是自然语言
5. 参考资料：https://blog.csdn.net/Mr_tyting/article/details/80091842

## GloVe - Stanford
**pip install glove_python**
## BERT
1. 参考资料：
    - https://zhuanlan.zhihu.com/p/49271699
    - https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650751075&idx=2&sn=0a3ecd1af5f8549051760775e34db342&chksm=871a841db06d0d0bcf3cc4e620bb384e050ba6e92224d338a8ddc1543add97a4a4e7919ebf15&scene=21#wechat_redirect
    - https://www.jiqizhixin.com/articles/2019-02-18-12?from=synced&keyword=NLP
## 数据：
1. 腾讯AI Lab开源中文词向量数据：https://ai.tencent.com/ailab/nlp/embedding.html
2. 中文维基百科数据： https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2
## 实例
NLP入门实例推荐（Tensorflow实现）https://blog.csdn.net/Irving_zhang/article/details/69396923
## 思路
1. 新词库
    - “”中的文字不拆分
    - 专有名词不拆分
    - “操作”词汇标注
2. 统一相似or模糊or错别字词语，如登陆，登入统一为登录 —— 中文输入纠错：https://blog.csdn.net/hqc888688/article/details/74858126
