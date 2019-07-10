---
tags: [nlp]
title: NLP & DL
created: '2019-07-10T02:04:58.746Z'
modified: '2019-07-10T06:15:08.610Z'
---

# NLP & DL
## 概览
深度学习代表nlp从机器学习到认知计算的进步。\
语言模型是一串词序列的概率分布。通过LM可以量化地评估一串文字存在的可能性。\
所有单词的概率乘积来评估文本存在的可能性-->简化的N元模型：对当前词的前N个词进行计算来评估该词的条件概率，N越大越容易出现数据稀疏问题，估算结果越不准。\
深度学习在NLP中的应用如下图所示：
![dl4nlp](https://github.com/lllssf/Fight-for-offer/blob/master/HW/NLP/DL4nlp.png)
## Basic Embedding Model
### NNLM
NNLM是2003年提出的一种基于神经网络的语言模型，基于词的分布式表示，学习词序列的概率函数。
### 分布式表达（Distributed Representation）
为解决语言模型的维数灾难问题，1986年Hinton提出了DR，将词表示成低维稠密实值向量。
#### 词向量
词向量是NLP深度学习研究的基石。
1. **CBOW**和**Skip-gram**是Word2vec的两种不同训练方式。CBOW指抠掉一个词，通过上下文预测该词；Skip-gram则与CBOW相反，通过一个词预测其上下文。
2. CBOW是一个只包含一个隐含层的全连接神经网络，输入层采用One-hot编码，输出层通过softmax函数得到词典里每个词的分布概率。
3. 基于字级别的特征表示丰富词向量信息《[Enriching Word Vectors with Subword Information](https://arxiv.org/pdf/1607.04606.pdf)》
4. Byte级别的特征表示方式BPE应用到机器翻译领域《[Cross-Lingual Language Model Pretraining](https://arxiv.org/pdf/1901.07291v1.pdf)》
5. EMLo、GPT和BERT等预训练模型结合上下文表示单词可以解决一词多义问题
6. 句向量Quick-Thought，文章向量doc2vec
### FastText
FastText是《[Bag of Tricks for Efficient Text Classification](https://arxiv.org/pdf/1607.01759.pdf)》提出的用于文本分类的深度学习模型，效果好速度快，原因是：
- 引入subword n-gram解决词态变化问题，利用字级别丰富单词信息；
- 用**基于霍夫曼树的分层softmax函数**将计算复杂度从O(kh)降低到O(hlog2(k))，其中k是类别数，h是文本表示的维数。

## CNN-based Model
1. 在词向量的基础上，需要一种有效的特征提取器从词向量序列里提取出更高层次的特征，应用到NLP任务中。
2. CNN有长距离依赖问题，现在的研究趋势是：加入**GLU/GTU门机制**来简化梯度传播，使用**Dilated CNN**增加覆盖长度，基于一维卷积层叠加深度并用**Residual Connections**辅助优化，优秀代表：增加网络深度的**VDCNN**和引进Gate机制的**GCNN**。

## RNN-based Model
1. CNN本质上通过卷积运算和池化操作提取关键信息，擅长捕获**局部特征**，而RNN则擅长处理时序信息和长距离依赖，各有专长，要具体问题具体分析选择哪一种模型。\
   RNN模型可用于语言模型和文本生成、机器翻译、语音识别、图像描述生成。
2. **LSTM\GRU**采用Gate机制解决了RNN的梯度消失问题,具体选择LSTM还是GRU，取决于其他因素，譬如计算资源等。
3. 在很多NLP任务场景，基于上下文的双层双向语言模型相比更有优势，能够更好地捕获变长且双向的n-gram信息，优秀代表**Bi-LSTM**，前沿的上下文预训练模型CoVe、ELMo、CVT等便是基于Bi-LSTM。RNN的模型特性使其很适合Seq2Seq场景，基于RNN/LSTM的Encoder-Decoder框架常用于机器翻译等任务。此外，还有尝试结合CNN和RNN的RCNN、基于RNN的文本生成模型VAE、结合LSTM和树形结构的解析框架Tree-base LSTM。
4. **CRF**(conditional random field,条件随机场)是常用于NER等序列标注任务的概率图模型。
5. **Seq2Seq**，即Sequence to Sequence，通用的Encoder-Decoder框架，常用于机器翻译、文本摘要、聊天机器人、阅读理解、语音识别、图像描述生成、图像问答等场景。

## Attention-based Model
1. 注意力机制是一种让模型重点关注重要关键信息并将其特征提取用于学习分析的策略。
2. 传统的Encoder-Decoder框架有个问题：一些跟当前任务无关的信息都会被编码器强制编码进去。而注意力机制对输入文本的每个单词赋予不同的权重，携带关键重要信息的单词偏向性地赋予更高的权重。抽象来说，即是：对于输入Input，有相应的query向量和key-value向量集合，通过计算query和key关系的function，赋予每个value不同的权重，最终得到一个正确的向量输出Output。
3. 注意力机制最早是为了解决Seq2Seq问题的，后来研究者尝试将其应用到情感分析、句对关系判别等其他任务场景，譬如关注aspect的情感分析模型ATAE LSTM、分析句对关系的ABCNN。

## Transformer-based Model
1. Google提出了Transformer，完全抛弃了CNN和RNN，只基于注意力机制捕捉输入和输出的全局关系，框架更容易并行计算，在诸如机器翻译和解析等任务训练时间减少，效果提升。
2. 论文提到了两个注意力机制：**Scaled Dot-Product Attention**和**Multi-Head Attention**。
3. **GPT**分成两个阶段：第一阶段采用Transformer解码器部分，基于无标注语料进行生成式预训练；第二阶段基于特定任务进行有区别地Fine-tuning训练，譬如文本分类、句对关系判别、文本相似性、多选任务。
4. **BERT**与GPT不同的是，BERT采用的特征提取器是Transformer编码器部分，句首用符号[CLS]表示，句尾用符号[SEP]表示，如果是句对，句子之间添加符号[SEP]。输入特征，由Token向量、Segment向量和Position向量三个共同组成，分别代表单词信息、句子信息、位置信息。
5. 2019年，Lample等在Facebook AI提出基于BERT优化的跨语言模型XLM《[Cross-Lingual Language Model Pretraining](https://arxiv.org/pdf/1901.07291v1.pdf)》，主要优化在于：
    - XLM模型采用一种**Byte-Pair Encoding（BPE）**编码方式，将文本输入切分成所有语言最通用的子单词，从而提升不同语言之间共享的词汇量
    - XLM模型的每个训练样本由两段内容一样但语言不通的文本组成，此外，每一种语言都会被随机掩盖单词
    - LM模型还会分别输入语言ID以及每一种语言的Token位置信息，这些新的元数据能够帮助模型更好地学习不同语言相关联Token之间的关系信息。

## 应用
自然语言处理有四大类常见的任务：
1. **序列标注**，譬如命名实体识别、语义标注、词性标注、分词等；
2. **分类任务**，譬如文本分类、情感分析等；
3. **句对关系判断**，譬如自然语言推理、问答QA、文本语义相似性等；
4. **生成式任务**，譬如机器翻译、文本摘要、写诗造句、图像描述生成等。













