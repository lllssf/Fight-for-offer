# 论文阅读笔记
## Attention
Tensorflow 实现在[Tensor2Tensor package](https://github.com/tensorflow/tensor2tensor)（相关文档在[T2T notebook](https://colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb)），哈佛大学团队用[PyTorch](http://nlp.seas.harvard.edu/2018/04/03/attention.html)实现了论文
### 阅读前参阅其他文章的笔记
- [Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention)](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
#### Seq2seq Models With Attention
1. sequence --> *encoder*(RNN) --> **context**(vector of floats, 大小一般是256,512,1024) --> *decoder*(RNN) -->sequence
2. encoder(RNN)输入：词嵌入向量 & 隐藏状态；输出：下个时间步的隐藏状态
    - words --> *Word Embedding* algorithm --> vectors,
    - 可以用[预训练词向量](https://github.com/Embedding/Chinese-Word-Vectors)也可以自己[训练词向量](https://blog.csdn.net/zhylhy520/article/details/87615772)
    - 大小一般是200-300维
3. classic decoder（RNN）输入：最后一个时间步的隐藏状态；输出：sequence\
   Attention decoder(RNN) 输入：所有隐藏状态；输出：sequence
    - 为所有隐藏状态赋予权重（*softmaxed score8），将权重加权输入给decoder
    - 权重训练是在decoder每个时间步完成的
    - 权重模型并不是简单的逐单词对应，例如下图
    ![score](https://jalammar.github.io/images/attention_sentence.png)
4. **Attention** allows the model to focus on the *relevant parts* of the input sequence as needed.
5. [Tensorflow实现](https://github.com/tensorflow/nmt)
#### Transformer
1. Input --> *encoder* --> *decoder* --> Output
2. encoder: Self-Attention --> Feed Forward NN\
   decoder: Self-Attention --> Encoder-Decoder Attention(如上文所述） --> Feed Forward NN
3. Self-Attention 使encoder在处理当前单词的同时利用了其他相关单词的处理信息。其实现步骤为：\
    i. 针对每个词向量创造三个向量：Query vector **Q**, Key vector **K**, Value vector **V**：
        - 每个向量由输入与其权重向量（**WQ,WK,WV**）相乘得到，权重向量在训练过程中更新
        - **q,k,v**的维度小于词向量，这种架构可以使Multi-Head Attention计算不变\
    ii.计算其他输入词向量相对于当前处理词向量的分值，这个分值决定我们在处理当前单词时要对其他输入单词投入多少注意力。**Score = q · v**
    ![score](https://jalammar.github.io/images/t/transformer_self_attention_score.png)\
    iii.分值除以Key vector的维度的平方根（在文中**k**的维度是64，这里除以了8），这一步为了使梯度更稳定。\
    iv.将上述得到的所有分值经过softmax运算。\
    v.用上一步得到的分值乘以每个单词的value vectors。\
    vi.对所有的value vectors加权求和得到当前处理单词的self-attention输出。\
    为提高self-attention的处理速度，该过程由矩阵运算完成。
    ![matrix](https://jalammar.github.io/images/t/self-attention-matrix-calculation-2.png)
4. 
### 阅读正文

## Bert

## XLNet
