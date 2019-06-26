# NLP任务攻坚筹备工作 
## 入门建议
1. **了解NLP最基本的知识**：《Speech and Language Processing》
2. **了解早年经典的NLP模型以及论文**：例如机器翻译中的IBM模型
3. **了解机器学习的基本模型**：吴恩达的 machine learning 
4. **多看NLP其他子领域的论文**：NLP 有很多子领域，MT，信息抽取，parsing，tagging，情感分析，MRC 等等
5. **了解 CV 和 data mining 领域的基本重大进展**
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
        - 语义角色标注（semantic role labeling）
        - 深层语义分析：语义依存分析（semantic dependency parsing）
    - 篇章级语义分析：
        - 篇章连接词识别
        - 论元识别
        - 显式篇章关系识别
        - 隐式篇章关系识别

![avatar](https://pic3.zhimg.com/v2-3d2cc9e84d5912dac812dc51ddee54fa_r.jpg)
## 任务流程：
1. 预训练 --> 中文分词 --> 词性标注 --> 依存句法分析
2. 方法：
    - 序列标注方法（Sequence Labeling）：给字打标签，很难利用词级别的信息
    - 基于转移的方法（Transition-based Method）：通过转移动作序列进行分词，即从左往右判断相邻的字是否分开。
    - 基于图的方法(Graph-based Method)
### 预训练
- 语言模型（Language Model，LM）：一串词序列的概率分布，通过概率模型来表示文本语义。
预训练是为了解决当前任务带有标注信息的数据有限问题，通过设计好的网络结构对大量自然语言文本抽取出大量语言学知识特征，这样就补充了数据不充足的当前任务的特征。
#### CV预训练
1. 浅层加载其他数据训练好的参数，然后：
    - Frozen：浅层参数在训练过程中不变
    - Fine-Tuning：浅层参数随训练过程改变
2. 为解决当前数据量小的问题，且能加快训练过程的收敛速度
3. 有效原因：对于层级结构的CNN，底层网络学习到的是可复用的特征，如线段等跟具体任务无关的通用特征
#### NLP预训练
1. **Word Embedding**是NLP里的早期预训练技术：通过神经网络学习语言模型任务过程中，Onehot编码作为原始单词输入，乘以矩阵Q获得向量C，然后用于预测单词。这个C就是单词对应的Word Embedding值，Q是网络参数需要学习，训练好后每一行代表一个单词对应的Word embedding值。这个网络不仅能够根据上下文预测后接单词，同时获得的副产品——矩阵Q。
2. 2013最火的语言模型做Word Embedding的工具是**Word2Vec**，训练方法分为：
    - CBOW：从一个句子里去掉一个词，用这个词的上下文预测被抠掉的词；
    - Skip-gram：输入某单词预测上下文单词
3. 18年以前，使用Word2Vec或Glove，通过做语言模型任务就可以获得每个单词的Word Embedding。此Q矩阵，即Word Embedding就是Onehot层到embedding层的网络预训练参数，后续训练过程也分为Frozen和Fine-Tuning。效果不显著的原因是**多义词问题**，同一单词经WordsVec会映射到同一word embedding值，无法区分不同语义。
4. 从Word Embedding到**ELMO（Embedding from language models）**，提出的论文为“*Deep contextualized word representation*”。ELMO根据当前上下文对Word Embedding进行动态调整。此类预训练方法被称为“*Feature-based Pre-Training*”。缺点是：
    - 特征抽取器选择的是LSTM（**LSTM提取长距离特征有长度限制**）而非Transformer（“*Attention is all you need*”提出，是个叠加的“自注意力机制（Self Attention）”构成的深度网络，是目前NLP里最强的特征提取器，参考资料：https://zhuanlan.zhihu.com/p/37601161 |可视化介绍：https://jalammar.github.io/illustrated-transformer/ | 代码及原理：http://nlp.seas.harvard.edu/2018/04/03/attention.html）
    - ELMO采取双向拼接比Bert一体化的融合特征方式弱。
5. 从Word Embedding到**GPT（Generative Pre-Training）**，GPT采用单向训练，会丢失context-after信息。GPT分两个阶段：
    - 使用Transformer特征抽取器利用语言模型进行预训练
    - 通过Fine-Tuning的模式解决下游任务，下游任务的网络结构要改造成和GPT的网络结构一样
6. NLP四大类任务：
    - **序列标注**：分词/词性标注/命名实体识别（Named Entity Recognition，NER）/语义标注……
    - **分类任务**：文本分类/情感计算……
    - **句子关系判断**：Entailment/QA/自然语言推理……
    - **生成式任务**：机器翻译/文本摘要……
7. **Bert（Bidirectional Encoder Representations from Transformers）**特点：*第一阶段采用双层双向特征抽取器Transformer通过MLM和NSP进行预训练*，*第二阶段采用Fine-Tuning模式应用到下游任务*。下游任务网络改造：对于序列标注问题，输入部分需要增加起始和终结符号，输出部分Transformer最后一次每个单词对应位置都进行分类。在构造双向语言模型方面：
    - **Masked Language Model（MLM）**本质思想是CBOW，细节方面有改进，利用mask掩码来标记被抠掉的词
    - **Next Sentence Prediction（NSP）**是指做预训练时要选择两个句子，一种是真正顺序相连的两句话，一种是第二句是随机选取拼接在第一句后的两句话。这种机制有助于句子关系判断任务。
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
- 中文标注集：863词性标注
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

### 依存句法分析 or 依存语义分析
**依存语法(Dependency Parsing, DP)** 通过分析语言单位内成分之间的依存关系揭示其句法结构。 直观来讲，依存句法分析识别句子中的“主谓宾”、“定状补”这些语法成分，并分析各成分之间的关系。依存句法分析标注关系 (共15种)
差别：
1. 句法依存某种程度上更重视非实词（介词）在句子结构中的作用，而语义依存更倾向具有直接语义关联的实词之间建立直接依存弧，非实词作为辅助标记。
#### 歧义消除


## 工具：
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
## word2vec - Google
```> pip install --upgrade gensim```
1. 词向量： One-hot representation --> Distributed Representation --> 计算欧氏距离或cos距离得到词语语义距离
2. 语言模型： N-gram, N-pos(part of speech) --> 拟合P
3. Word2vec有两种类型，每种类型有两种策略
4. CBOW加层次的网络结构：输入层（词向量）-->隐藏层（累加和）-->输出层（霍夫曼树）——判断一句话是否是自然语言
5. 参考资料：https://blog.csdn.net/Mr_tyting/article/details/80091842
## GloVe - Stanford
```> pip install glove_python```
## BERT - Google
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
## 数据：
1. 腾讯AI Lab开源中文词向量数据：https://ai.tencent.com/ailab/nlp/embedding.html
2. 中文维基百科数据： https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2
## 实例
NLP入门实例推荐（Tensorflow实现）https://blog.csdn.net/Irving_zhang/article/details/69396923
## 训练中文词向量
- Word2vec https://blog.csdn.net/lilong117194/article/details/82849054
- BERT 
        - https://blog.csdn.net/zhylhy520/article/details/87615772
        - https://blog.csdn.net/qq_29660957/article/details/88683823
       
## 思路
1. 新词库
    - “”中的文字不拆分
    - 专有名词不拆分
    - “操作”词汇标注
2. 统一相似or模糊or错别字词语，如登陆，登入统一为登录 —— 中文输入纠错：https://blog.csdn.net/hqc888688/article/details/74858126

## 任务目标
1.动宾短语提取
