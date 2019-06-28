# Speech and Language Processing(3rd) ——《自然语言处理综论》(1st)
## 导论
1. 状态空间搜索（state space search）和动态规划（dynamic programming）
2. 符号派（symbolic） V.S. 随机派（stochastic）
3. Brown美国英语语料库
4. 四个范本（paradigm）： 
  - 随机范型（stochastic paradigm）--HMM & 噪声信道与解码模型；
  - 逻辑范型（logic-based paradigm）--Q-system & 变形文法（metamorphosis grammar）
## 词汇的计算机处理
### 正则表达式（Regulation Expression，RE）
正则表达式（Regular Expression，RE）是一种用于描述文本搜索符号串的语言。
#### 基本的正则表达式模式
1. 可以是单独的字符如/!/，也可以是字符序列/urgl/
2. 区分大小写（case sensitive）。
3. “/[]/”内的字符是“或”的关系（析取，disjunction），/[1234567890]/检索的就是任意**一个**数字
4. “-”表示的是范围，/[0-9]/就是任意一个数字
5. “^”在[]中第一个位置时表示的是“非”，在[]中其他位置表示的就是该符号本身，/[^A-Z]/检索的就是第一个非大写字母的字符；还是一种**Anchor**，表示一行的开始，/^The/只检索一行开始的“The”
6. “?”表示前一个字符或有或无，/colou?r/检索的是‘color’和‘colour’
7. “\*”（Kleene star）表示前一个字符或RE出现**0次**或**连续**出现若干次，/a\*/检索的是多个a（aaa）和不是a的字符串。
8. “+”（Kleene +）表示前一个字符或RE出现**1次**或**连续**出现若干次，/[0-9]+/是数字序列的规范表达式。
9. “.”是除回车符之外的任意单字符的匹配符号，常与“\*”连用匹配任意字符串。
10. “$”也是一种Anchor，表示一行的结尾。/^The dog\.$/检索一行只包含“The dog.”
11.“\b”表示词的边界(word boundary)，而word被定义为数字、下划线或字母的任何序列。/\b99/不能检索到“299”，但可以检索到“￥99”。'er\b' 可以匹配"never" 中的 'er'，但不能匹配 "verb" 中的 'er'。'er\B' 能匹配 "verb" 中的 'er'，但不能匹配 "never" 中的 'er'。
#### 析取、组合与优先关系（Disjunction, Grouping, and Precedence）
1. “|” 析取符号disjunction operator，/cat|dog/检索“cat”或“dog”。
2. “()” 优先级最高。/gupp(y|ies)/检索的是“guppy”或“guppies”。RE中符号**优先层级**依次为:\
    “()” > 计数符“* + ? {}” > 序列和anchors “a ^start end$ \b” > “|”
3. RE是一般贪心模式，试图覆盖尽可能长的符号串；但 **“\*?”和“+?”** 是non-greedy，总是尽可能匹配最短的序列
4. 要避免**false positives**，如想检索“the”却检索到“other”，即提高精确率（**precision**），precision = TP/(TP+FP)；\
   要避免**false negatives**，例如漏检索了“The”，即提高召回率（**recall**）,recall = TP/(TP+FN)
#### 高级算符
1. “{}”计数符，/a\.{3}z/检索的是“a...z”

RE | Match |
:-:|-
\* | 前一个字符出现0或多次 |
\+ | 前一个字符出现1或多次 |
? | 前一个字符出现0或1次 |
{n} | 前一个字符出现n次 |
{n,m} | 前一个字符出现n到m次 |
{n,} | 前一个字符出现至少n次 |
{,m} | 前一个字符出现至多m次 |
2. 通用字符集的替换名

RE | Expansion | Match
:-:|-|-
\d | [0-9] | 任意一个数字
\D | [^0-9] | 一个非数字字符
\w | [a-zA-Z0-9_] |  任何一个字母数字下划线
\W | [^\w] | 非词
\s | [ \r\t\n\f] | 各种空
\S | [^\s] | 一个非空字符
\A | - | 匹配字符串开始
\Z | - | 匹配字符串结束，如果是存在换行，只匹配到换行前的结束字符串
\z | - | 匹配字符串结束
\G | - | 匹配最后匹配完成的位置
#### [Python3 正则表达式](https://www.runoob.com/python3/python3-reg-expressions.html#flags)
```
# 匹配函数，只匹配字符串的开始
re.match(pattern, string, re.M|re.I)
# 搜索函数，匹配整个字符串直到找到一个匹配
re.search(pattern, string, re.M|re.I)
# 检索和替换
re.sub(pattern, replace, string, count=0, flag=0)
# 编译生成正则表达式对象（pattern）
re.compile(pattern)
# 找到RE所匹配的所有子串
findall(string[, pos[, endpos]])
# 在字符串中找到正则表达式所匹配的所有子串，并把它们作为一个迭代器返回
re.finditer(pattern, string, flags=0)
# 分割字符串
re.split(pattern, string[, maxsplit=0, flags=0])
```
