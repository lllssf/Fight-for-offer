# 论文阅读笔记
## A Primer on Neural Network Modelsfor Natural Language Processing
神经网络应用于NLP的情况
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
3. **Self-Attention** 使encoder在处理当前单词的同时利用了其他相关单词的处理信息。其实现步骤为：\
    i. 针对每个词向量创造三个向量：Query vector **Q**, Key vector **K**, Value vector **V**：
        - 每个向量由输入与其权重向量（**WQ,WK,WV**）相乘得到，权重向量在训练过程中更新
        - **q,k,v**的维度小于词向量，这种架构可以使Multi-Head Attention计算不变\
    ii. 计算其他输入词向量相对于当前处理词向量的分值，这个分值决定我们在处理当前单词时要对其他输入单词投入多少注意力。**Score = q · v**
    ![score](https://jalammar.github.io/images/t/transformer_self_attention_score.png)\
    iii. 分值除以Key vector的维度的平方根（在文中**k**的维度是64，这里除以了8），这一步为了使梯度更稳定。\
    iv. 将上述得到的所有分值经过softmax运算。\
    v. 用上一步得到的分值乘以每个单词的value vectors。\
    vi. 对所有的value vectors加权求和得到当前处理单词的self-attention输出。\
    为提高self-attention的处理速度，该过程由矩阵运算完成。
    ![matrix](https://jalammar.github.io/images/t/self-attention-matrix-calculation-2.png)
4. 该论文进一步细化了self-attention通过加入**Multi-Head Attention**机制。文中共有8个attention heads，输出得到8个不同的Z，而FFNN的输入是一个词向量，所以将\[Z0 Z1 ··· Z7]乘以权重**WO** 
![W0](https://jalammar.github.io/images/t/transformer_attention_heads_weight_matrix_o.png)
5. 为了确定每个单词在输入序列中的位置，在每个embedding上又加入了一个向量**positional encoding vectors**，该向量遵循模型学习的特定模式，其中每个值在\[-1,1]区间。用TensorFlow实现是用[ get_timing_signal_1d()](https://github.com/tensorflow/tensor2tensor/blob/23bd23b9830059fbc349381b70d9429b5c40a139/tensor2tensor/layers/common_attention.py)，该实现的优点是能够根据未知的序列长度进行缩放（例如，在用训练好的模型翻译比训练集中任意句子都长的句子时）。
6. 使用了**残差结构**，如图所示：
![residual](https://jalammar.github.io/images/t/transformer_resideual_layer_norm_2.png)
7. 最顶层编码器的输出被转换为Attention vectors **K**和**V**，然后作为解码器“**encoder-decoder attention**”层的输入，而**Q**通过下层解码器的输出获取。
8. 每个时间步的输出会被反馈给下一个时间步作为输入，并像编码器一样会在每个词向量基础上添加positional vector。直到输出<EOS>结束输出。
![output](https://jalammar.github.io/images/t/transformer_decoding_2.gif)
9. 与编码器不同的是，解码器的self-attention层只关注输出序列中之前的位置，未输出位置被设置为-inf来屏蔽。
10. 编码器输出的是词向量，还要用Linear层（全连接层）和Softmax层来得到对应的单词，softmax的输出与“输出单词词典”一一对应。
11. **训练**：
    - 对输出词典进行one-hot编码，维度为词典长度
    - greedy decoding(保留概率最高的输出） V.S. beam search（保留概率最高的前n个输出，n是超参数）

### 阅读正文

## Bert

## XLNet
