## 知识图谱的搭建的简单流程
搭建知识图谱系统的核心在于对业务的理解以及对知识图谱本身的设计\
完整流程包括：定义具体的业务问题、数据的收集&预处理、 知识图谱的设计、把数据存入知识图谱、上层应用的开发以及系统的评估。
1. 定义具体的业务问题：重点在于对于自身的业务问题到底需不需要知识图谱系统的支持。
![no1](https://github.com/lllssf/Fight-for-offer/blob/master/DeeCamp/Knowledge%20graphs/image1.png)
2. 数据收集 & 预处理：对于垂直领域的知识图谱来说，它们的数据源主要来自两种渠道：一种是业务本身的数据，这部分数据通常包含在公司内的数据库表并以**结构化**的方式存储；另一种是网络上公开、抓取的数据，这些数据通常是以网页的形式存在所以是**非结构化**的数据。后者需要借助于NLP等技术，如：实体命名识别、关系抽取、实体统一、指代消解等。
3. 知识图谱的设计：设计原则--BAEF原则：
    - 业务原则（**B**usiness Principle）：一切要从业务逻辑出发，好的设计很容易让人看到业务本身的逻辑，要想好未来业务可能的变换
    - 分析原则（**A**nalytics Principle）：不需要把跟关系分析无关的实体放在KG当中
    - 效率原则（**E**fficiency Principle）：KG尽量轻量化，把常用的信息存放在知识图谱中，把那些访问频率不高，对关系分析无关紧要的信息放在传统的关系型数据库当中
    - 冗余原则（**R**edundancy Principle）：有些重复性信息、高频信息可以放到传统数据库当中
4. 数据存入知识图谱：知识图谱主要有两种存储方式：RDF & 属性图，Neo4j系统目前是使用率最高的图数据库，Jena是RDF的存储系统
![cunchu](https://image.jiqizhixin.com/uploads/editor/6d23ee75-a606-46ac-9e00-3320a870c0e8/1529464461822.png)
5.上层应用的开发：从算法的角度来讲，有两种不同的场景：一种是基于规则的；另一种是基于概率的。
   - 基于规则的方法论：
      - 不一致性验证：判断网络中存在的风险，通过规则找出潜在的矛盾点
      - 基于规则的特征提取：基于深度搜索的涉及深度的关系的特征
      - 基于模式的判断：通过一些模式来找到有可能存在风险的子图（sub-graph），然后对这部分子图做进一步的分析
    - 基于概率的方法：社区挖掘、标签传播、聚类等算法；需要足够多的数据！
    - 基于动态网络的分析：KG结构随时间变化的分析
## 论文
1. 综述性论文
## 参考资料
1. [知识图谱综述#1: 概念以及构建技术](https://mp.weixin.qq.com/s/bhk6iZdphif74HJlyUZOBQ?)
2. [知识图谱综述#2: 构建技术与典型应用](https://mp.weixin.qq.com/s/j1ub_exp-T7kk7snHs4eYw)
3. [当知识图谱“遇见”深度学习](https://blog.csdn.net/heyc861221/article/details/80129309)

Paper 
Extracting Multiple-Relations in One-Pass with Pre-Trained Transformers
Joint entity recognition and relation extraction as a multi-head selection problem
 
permID service: 做实体去重的服务，拿到唯一实体ID
https://permid.org/
download entity data: 做实体去重的服务，拿到唯一实体ID
https://permid.org/download
Import to Neo4J guide:  导入实体到Neo4J 图数据库的步骤
https://developers.refinitiv.com/knowledge-graph/knowledge-graph-feed-api/learning?content=48179&type=learning_material_item
 
Neo4j guide: Neo4J图数据库的基本操作使用手册
https://neo4j.com/developer/get-started/
 
Cypher query language:  图数据库的查询语言
https://neo4j.com/developer/cypher/
