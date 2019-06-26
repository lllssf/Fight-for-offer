# Transformer 理解
参考：
- [Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention)](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
## Seq2seq Models With Attention
1. sequence --> *encoder*(RNN) --> **context**(vector of floats, 大小一般是256,512,1024) --> *decoder*(RNN) -->sequence
2. encoder(RNN)输入：词嵌入向量 & 隐藏状态
  - words --> *Word Embedding* algorithm --> vectors,
  - 可以用[预训练词向量](https://github.com/Embedding/Chinese-Word-Vectors)也可以自己[训练词向量](https://blog.csdn.net/zhylhy520/article/details/87615772)
  - 大小一般是200-300维
