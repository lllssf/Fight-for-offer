# NLP 
## 语言底层分析技术：
### 包括：分词、词性标注、命名实体识别、句法分析
### 难点： 
    - 语言使用灵活
    - 书写错误，纠错成为必要的预处理环节
    - 新词
### 中文分词
1. 将连续的字序列按照一定的规范切分成词序列
2. 中文分词准确率低的原因：
    - 词表收录
    - 分词规范
    - 新词识别
    - 歧义切分
3. 中文分词工具： jieba, SnowNLP, PkuSeg, THULAC, HanLP
### 词语相似度
1. 基于世界知识（Ontology），一般利用同义词词典等 -> 计算词语语义距离
2. 基于大规模的语料来统计， -> 词语的相关性
3. 《知网》是一个以汉语和英语的词语所代表的概念为描述对象，以揭示概念与概念之间以及概念所具有的属性之间的关系为基本内容的常识知识库。：
    - 概念：对词汇语义的描述
    - 义原：知识表示语言（描述一个”概念“的最小意义单位），《知网》共有1500义原

## 正则表达式

## HMM（隐马尔科夫模型）
### Introduction
寻找一个事物在一段时间里的变化模式（规律）
### 生成模式（Generating Patterns）
1. 确定性模式（Deterministic Patterns）
2. 非确定性模式（Non-deterministic Patterns）：
    - **马尔可夫假设**：模型的当前状态仅仅依赖于前面的几个状态
    - 状态转移概率并不随时间变化而不同（常常不符合实际）。
    - **pi向量**： 定义系统初始化时每个状态的概率。
3. 隐藏模式（Hidden Patterns）
    - **隐马尔可夫模型（Hidden Markov Models）**：在一个标准的马尔可夫过程中引入一组观察状态以及其与隐藏状态的概率关系。
    - **隐藏状态**（一个系统的真实状态，可由Markov过程描述） --> **观察状态**（可直接观测的状态）
    - **pi向量**： 隐藏状态的初始概率
    - **状态转移矩阵**：隐藏状态->另一个隐藏状态的概率
    - **混淆矩阵（confusion matrix）**：隐藏状态->观察状态的概率。
### 隐马尔科夫模型（HMM）
1. 定义：三元组(pi, A, B)
    - pi：初始概率向量
    - A：状态转移矩阵
    - B：混淆矩阵
2. 应用：
    - 评估：**前向算法（Forward algorithm）**对一个观察序列匹配最可能的HMM —— 语音识别
    - 解码：**Viterbi算法**搜索已知观察序列及HMM情况下最可能的隐藏状态序列 —— 词性标注（观察状态-句子中的单词，隐藏状态-词性）
    - 学习：**前向-后向算法**被用来进行参数估计（A和B不能被直接测量），根据观察序列和与其有关的隐藏状态集来估计最合适的HMM。
3. 前向算法：
    - 穷举搜索
    
## Paper: CNN for Sentence Classification
1. filter: h*k

## 书籍：Speech and Language Processing(3rd)
https://web.stanford.edu/~jurafsky/slp3/

## word2vec - Google
**pip install --upgrade gensim**
1. 词向量： One-hot representation --> Distributed Representation --> 计算欧氏距离或cos距离得到词语语义距离
2. 语言模型： N-gram, N-pos(part of speech) --> 拟合P
3. Word2vec有两种类型，每种类型有两种策略
4. CBOW加层次的网络结构：输入层（词向量）-->隐藏层（累加和）-->输出层（霍夫曼树）——判断一句话是否是自然语言

## GloVe - Stanford
**pip install glove_python**

## 数据：
1. 腾讯AI Lab开源中文词向量数据：https://ai.tencent.com/ailab/nlp/embedding.html
2. 中文维基百科数据： https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2

## 路线
1. 新词库
    - “”中的文字不拆分
    - 专有名词不拆分
