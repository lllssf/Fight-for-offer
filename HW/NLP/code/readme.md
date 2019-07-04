1. data_process（部分）:\
   数据处理，只保留有效文字信息。
2. sentence_seg:\
   分句，‘’“”【】[]''中的内容不能分割，1.balabala格式分割，逗号也分割
   ```
   text = '1.玄德幼时，与乡中小儿戏于树下，曰：“我为天子，当乘此车盖。”2.行吧，【唉呀妈呀，好的吧】'
   s = sentence_seg(text)
   print('\n'.join(s))

   OUTPUT:
   1.玄德幼时，
   与乡中小儿戏于树下，
   曰：“我为天子，当乘此车盖。”
   2.行吧，
   【唉呀妈呀，好的吧】
   ```
3. 使用Bert计算中文词向量余弦相似度
   - 下载训练好的中文[BERT模型](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)
   - 使用肖涵博士提供的[bert-as-service](https://github.com/hanxiao/bert-as-service)
     ```
     # 先在命令行运行
     >pip install  bert-serving-server
     >pip install bert-serving-client
     >D:\python36\Scripts>bert-serving-start -model_dir D:\Bert\chinese_L-12_H-768_A-12
     ```
4. 用pyltp分词和词性标注构建词典，速度慢，建议使用其他工具（使用jieba速度快很多，代码类似）。
5. 用BERT将中文词语转化为词向量再使用K-Means聚类将中文词语划分为不同的簇。
   因为K-means需要提供分类数量所以不够有效，待改进。

   
