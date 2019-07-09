## 综述性论文
### 《知识表示学习研究进展》
1. 表示学习：通过机器学习将研究对象的语义信息表示为**稠密低维实值向量**\
            本质特点：**分布式表示，层次结构**。
2. 知识表示学习是面向知识库中实体和关系的表示学习。
3. 知识表示学习的代表模型：
    - 距离模型：结构表示（structured embedding，SE）将知识库三元组作为学习样例，找出让头、尾实体距离最近的关系矩阵。
    - 单层神经网络模型（single layer model，SLM）：在SE基础上加入了非线性操作
    - 语义匹配能量模型（semantic matching energy，SME）
    - 隐变量模型（latent factor model，LFM）利用基于关系的双线性变换刻画实体和关系之间的二阶联系。
    - 张量神经网络模型（neural tensor network，NTN）利用双线性张量取代传统神经网络中的线性变换层。
    - 矩阵分解模型：RESACL模型
    - 翻译模型：TransE模型
    - 全息表示模型（holographic embeddings，Hole）使用头、尾实体向量的“循环相关”操作来表示该实体对
### 《知识图谱研究进展》
1. 知识图谱本质上是一种叫做**语义网络（semantic network）**的知识库，即具有有向图结构的一个知识库，其中图的结点代表**实体（entity）**或**概念（concept）**，边代表语义关系。
2. 从非结构化数据中获取知识：正文提取-->实体识别（分词，词性标注，词向量，关键词提取，主题获取）-->实体关系识别（依存分析，语义解析，语义角色标注）
3. 从半结构化数据中获取知识：少量标注数据-->使机器学习半结构化数据的规则-->使用规则对同类型或符合某种关系的数据进行抽取-->当用户的数据存储在生产系统的数据库中时，需要通过**ETL 工具**对用户生产系统下的数据进行重新组织、清洗、检测最后得到符合用户使用目的数据。
4. 多源知识融合需要统一的术语，而提供统一术语的结构或者数据被称为**本体**，本体不仅提供了统一的术语字典，还构建了各个术语间的关系以及限制。
5. **知识融合(knowledge fusion)** 包含了数据映射技术、实体匹配技术、本体融合技术、如NoSQL或关系数据库等的存储架构和如Spark、Hadoop的大数据平台的应用。
6. 知识图谱技术地图 
![map](https://github.com/lllssf/Fight-for-offer/blob/master/DeeCamp/Knowledge%20graphs/papers/KGmap.JPG)
7. 开放知识图谱：
    - [DBpedia](https://wiki.dbpedia.org/): 可视为维基百科的结构化版本
    - [Yago](https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/yago-naga/yago/downloads/)：整合了维基百科与WordNet的大规模本体
    - [Wikidata](https://www.wikidata.org/wiki/Wikidata:Main_Page): 可以自由协作编辑的多语言百科知识库
    - [BabelNet](https://babelnet.org/): 目前世界范围内最大的多语言百科同义词典，它本身可被视为一个由概念、实体、关系构成的语义网络
    - [ConceptNet](http://www.conceptnet.io/): 一个大规模的多语言常识知识库，其本质为一个以自然语言的方式描述人类常识的大型语义网络
    - [Microsoft Concept Graph](https://concept.research.microsoft.com/): 一个大规模的英文Taxonomy，其中主要包含的是概念间以及实例（等同于上文中的实体）概念间的IsA 关系
    - 中文开放知识图谱联盟[OpenKG](http://www.openkg.cn/): 包含91个数据集，15 个类目的开放知识图谱。
