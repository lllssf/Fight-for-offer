# 720
## 让人工智能造福人类——李开复
1. 正确的时间做正确的事情；
2. 当前很多业务的海量数据依靠于客观、精准的自然标注，如购买行为等。
3. 关于隐私问题要找到保护隐私和使用技术便利性的trade off：
   - 使用技术保护隐私：联邦学习、同态加密
   - 制定相应法规、防止数据滥用
4. 关于对AI的非良性使用，如DeepFake，还是要靠技术解决技术问题，以后如何辨别fake和真实也是一个方向
5. 关于AI是否会取代人类工作的问题：一些岗位的消失也会带来新岗位的出现或增多，如滴滴、Uber带来的快车司机岗位。
6. 5G + IOT，边缘计算。
7. 工作中更注重商业价值，寻找最快达到目的的途径。
8. 对传统任务的AI赋能是第三波潮流。
## AI教育与教育AI——俞勇
知识追踪：
        - 贝叶斯知识追踪
        - 因子分析知识追踪
        - 深度知识追踪（LSTM）
        - 深度记忆网络（DKWMN）
        - 生存分析
## 机器学习现状与进展——张潼
### 机器学习发展现状
1. 大数据 + 大算力 + 深度学习
2. 现状：单点能力超越人类
3. 大数据与大算力为**复杂模型、向量表示、自动化机器学习、基于模拟的强化学习**做出了贡献
4. 现有问题：
   - 鲁棒性不强
   - 场景变动带来的自适应性不强
   - 任务可迁移性不强
   - 对世界理解不够；没有好的知识表示方法
5. 单一模型的多任务自主学习
### 机器学习前沿方向和进展
#### 复杂模型
1. Deep ==> - 更高效地表示复杂函数；深度网络学习高级特征
2. 训练的Tricks缺乏足够、合理的理论支撑。如BatchNormalization，目前的观点是：Intern Covariance shift & Increase smoothness
3. ResNet：relation to ODE（Ordinary Differential Equation）
4. Overparameterized DNN can be efficiently learned
5. DNN with Geometric structure：CNN、RNN、GNN……
6. Local --> Gloabal: Attention; Transformer
#### 表示学习
1. 目标： data --> vector/embedding ( Dense vector, 可迁移）
2. 无监督、半监督、有监督学习
3. 无监督：AutoEncoder, word2vec, context sensitive word2vec, Application Small sample learning, Bert
4. AutoEncoder --> Variational AutoEncoder(VAE): 可以用来生成高质量图片
5. word2vec --> GPT --> BERT
6. Small sample learning:
   - Matching Networks: [Matching Networks for One Shot Learning](https://arxiv.org/pdf/1606.04080.pdf)
   - Prototypical Networks: [Prototypical Networks for Few-shot Learning](https://arxiv.org/pdf/1703.05175.pdf)
#### AutoML
1. Data Augmentation
2. Neural Architecture Search(NAS): using RL , 是自动设计网络的有效great tool
3. Efficient Net: Rethinking model scaling for CNN
## 机器学习的挑战——周志华
### 关于深度模型
1. Deep Learning =? DNN （当前很多人是这么认为的）
2. activation function：连续可微（是BP work的前提，可微是NN的基础）
3. 2006年以前，因为梯度消失，5层以上的网络都训练失败。算法的设计与改进&大算力使得Deep
4. Why Deep？ 
   - Increase model complexity --> Increase learning ability
   - 变宽是神经单元个数的增加，变深是非线性函数的嵌套更复杂。
5. **BUT**，deep也会导致increase risk of overfitting & diificulty in training 
6. 解决：big training data , powerful campulation facilities, Training tricks ==> high-complexity --> DNNs
7. 1989年时就证明两层的神经网络就可以拟合任意函数，那么， Why "falt" not good?
8. 表示学习的关键：**逐层处理（layer-by-layer processing）**。如果仅有逐层处理呢？ --> 决策树？Boosting？（x）：复杂度不足，始终基于原始特征，无特征变换。 So， 逐层处理 + 特征变换 ==> 深度模型 --
