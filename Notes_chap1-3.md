# 第一章 新手上路

自然语言处理涉及了计算机科学、语言学、人工智能。

## 1.1 自然语言与编程语言

- 自然语言词汇量丰富（新词汇也多）
- 自然语言是非结构化的，而编程语言有明确的结构关系。=》计算机理解自然语言会更困难
- 自然语言有大量歧义
- 自然语言容错性高，编程语言在编译等时候会有一些警告和错误。=》自然语言的规范性得不到保证
- 自然语言的变化更迅速、更嘈杂，编程语言的变化要缓慢、温和一些。
- 自然语言有一定的简略性，根据一些习惯省略了东西（如简称、代词），省略掉的常识会给处理带来障碍

## 1.2 NLP层次

![202210211148572](https://cdn.jsdelivr.net/gh/Ga7777/Notes_NLPbook/imgs/202210211237656.png)

1. 输入源：语言、图像和文本（转化成文本）

2. 对词语层次的词法分析：

   **中文分词**（将文本分割成有意义的词语） e.g. 先  有  通货膨胀  干扰

   **词性标注**（确定每个词语的类别和浅层的歧义消除）e.g.迈向/v 充满/v  希望/n  的/u  新/a  世纪/n —— /w

   **命名实体识别**（识别出较长的专有名词，包括任命、地名、机构名等）e.g. 萨哈夫/nr 说/v，/w伊拉克/ns将/d

   词法嵌套会增加难度，好多复合词，比如 联合国销毁伊拉克大规模杀伤性武器特别委员会

3. 信息抽取、文本聚类、文本分类、**句法分析** （更关注语法）

4. **语义分析**（更关注语义）、篇章分析

5. 其他高级任务如信息检索（IR）等。（IR的目标是查询信息，NLP是理解语言）

## 1.3 NLP流派

1. 1980年前：基于规则的专家系统：人力考虑所有情况，矛盾的地方通过优先级来解决。最大的弱点：难以拓展。（手写规则、面向逻辑推理的编程语言）

2. 1980年-2010年，基于统计的学习方法：让计算机自动学习语言。统计是在语料库上统计，需要制作语料库。这也需要专家们根据语言学知识未统计模型设计**特征模板**。更自适应、灵活。（大量机器学习的理论与应用、革命性实用水准）

   1988年，隐马尔科夫模型

   1995年，第一个健壮的基于统计的句法分析器

   2000年，感知机、条件随机场

3. 基于深度学习的方法：神经网络的复兴、表示学习、端到端的设计

一些较好的模型和论文准确率对比：

![image-20221017172135447](https://cdn.jsdelivr.net/gh/Ga7777/Notes_NLPbook/imgs/202210211158014.png)

## 1.4 机器学习

**监督学习**：非结构化数据标注——标注后的结构化数据去训练得到模型——非结构化数据输入进模型——预测得到结构化数据结果。

**无监督学习**：聚类、降维。机器只能发现样本之间的联系，无法学习样本与答案的联系。

**半监督学习**：训练多个模型，对一个实例预测后将多数一致的结果扩充进训练集。这样可以综合利用标注数据和丰富的未标注数据。*<热门>*

**强化学习**：边预测，边根据结果反馈规划下次决策。有因果、彼此关联。（在人机交互问题上成果斐然）

## 1.5 语料库建设

规范制定：注集定义、样例和实施方法。

人员培训：尤其是多人协同

人工标注：brat 词性标注、命名实体识别、句法分析等。



**<!--开源工具已配置完成。-->**



# 第二章 词典分词

常用的分词算法。匹配规则不难，难的是效率与内存平衡。

齐夫定律：一个单词的词频和它的词频排名成反比。

## 2.1 切分算法

词典分词规则没有技术含量，消歧效果差。其核心价值不在精度而在速度。双向的速度约是单向的一半。

- **完全切分**：遍历，找出所有单词（无意义）

```python
def fully_segment(text, dic):
    word_list = []
    for i in range(len(text)):                  # i 从 0 到text的最后一个字的下标遍历
        for j in range(i + 1, len(text) + 1):   # j 遍历[i + 1, len(text)]区间
            word = text[i:j]                    # 取出连续区间[i, j]对应的字符串
            if word in dic:                     # 如果在词典中，则认为是一个词
                word_list.append(word)
    return word_list
```

```python
fully_segment('商品和服务', dic) #['商', '商品', '品', '和', '和服', '服', '服务', '务']
```

- **正向最长匹配**：单词越长，优先级越高

```python
def forward_segment(text, dic):
    word_list = []
    i = 0
    while i < len(text):
        longest_word = text[i]                      # 当前扫描位置的单字
        for j in range(i + 1, len(text) + 1):       # 所有可能的结尾
            word = text[i:j]                        # 从当前位置到结尾的连续字符串
            if word in dic:                         # 在词典中
                if len(word) > len(longest_word):   # 并且更长
                    longest_word = word             # 则更优先输出
        word_list.append(longest_word)              # 输出最长词
        i += len(longest_word)                      # 正向扫描
    return word_list
```

```python
forward_segment('研究生命起源', dic) #研究生，命，起源
```

- **逆向最长匹配**

```python
def backward_segment(text, dic):
    word_list = []
    i = len(text) - 1
    while i >= 0:                                   # 扫描位置作为终点
        longest_word = text[i]                      # 扫描位置的单字
        for j in range(0, i):                       # 遍历[0, i]区间作为待查询词语的起点
            word = text[j: i + 1]                   # 取出[j, i]区间作为待查询单词
            if word in dic:
                if len(word) > len(longest_word):   # 越长优先级越高
                    longest_word = word
                    break
        word_list.insert(0, longest_word)           # 逆向扫描，所以越先查出的单词在位置上越靠后
        i -= len(longest_word)
    return word_list
```

```python
backward_segment('研究生命起源', dic) #研究，生命，起源
```

- **双向最长匹配**：选词数更少的——否则选单字更少的——否则用逆向最长匹配（是统计规律，有数据支撑）

```python
def bidirectional_segment(text, dic):
    f = forward_segment(text, dic)
    b = backward_segment(text, dic)
    if len(f) < len(b):                                  # 词数更少优先级更高
        return f
    elif len(f) > len(b):
        return b
    else:
        if count_single_char(f) < count_single_char(b):  # 单字更少优先级更高
            return f
        else:
            return b                                     # 都相等时逆向匹配优先级更高

backward_segment('研究生命起源', dic) #研究，生命，起源
```

## 2.2 字典树

### 2.2.1 Trie树

- **提出动机**：匹配算法的瓶颈在于判断集合中是否有字符串。Treemap复杂度为O(logn)，Hashmap内存复杂度太大。需要一种***速度又快、又省内存***的方式。
- **基础信息**：蓝色节点标记结尾。空间利用率较低。<!--字典树其实就是确定有限状态自动机DFA。-->
- **性能分析**：当词典大小为n时，最坏情况下字典树的复杂度是***O(logn)***（假设词语都是单字，且子节点用对数复杂度的数据结构存储），但实际速度比二分查找快。因为**前缀匹配是递进的过程**，算法不用比较字符串的前缀。

![image-20221019201953339](https://cdn.jsdelivr.net/gh/Ga7777/Notes_NLPbook/imgs/202210211158015.png)

### 2.2.2 BinTrie树（首字散列）

- **提出动机**：Trie树速度慢，空间利用率低
- **第一步修改**：仅在根节点散列，其余二分。

![image-20221019202020490](https://cdn.jsdelivr.net/gh/Ga7777/Notes_NLPbook/imgs/202210211158016.png)

速度快了将近一倍，但仍有不足——没有发挥出前缀的妙用。

- **修改动机**：朴素接口实现时，一次查询自、自然、自然语、自然语言是否在词典中。但事实上因为共同前缀的问题，若自然不在字典中，那么自然语、自然语言也一定不在。那么状态转移失败时（自向自然），可以提前终止对自开头的扫描。
- **第二步修改**：为了发挥前缀的作用，设置了BinTrie接口，以全切分为例：

<!--state：当前状态-->

<!--bigin：当前扫描起点-->

<!--i：状态转移时接受字符的下标-->

<!--processor：回调函数-->

<!--从根节点this开始顺序选择起点，然后递增i进行状态转移（if分支）；一旦状态转移失败（else分支），对以begin开头的词语扫描立即停止，bigin递增，最后重新开始新前缀的扫描。-->

![image-20221019202328356](https://cdn.jsdelivr.net/gh/Ga7777/Notes_NLPbook/imgs/202210211158017.png)

e.g. 

从”自“到”自然“，若无词条，state==null，

进入else，状态转移失败，那么对以begin开头的其他扫描立即停止（所有自然语言、自然人等词都不存在），

重新进入对新前缀的搜索（以''然"开头来搜索）。

- **性能分析**：比朴素接口速度快很多，将BinTrie原生接口做到了1000万字/秒的速度，比python的64万字/秒提高了两个数量级。但除了根节点完美散列外，其余节点都在二分查找，当存在c个子节点时，每次状态转移复杂度为***O(logc)***.

### 2.2.3 DAT（双数组字典树）

- **提出动机**：当存在c个子节点时，Bintrie树每次状态转移复杂度为***O(logc)***。当c很大时依然很慢。DAT的状态转移复杂度为***常数***，由base和check数组构成。
- **基础知识：**

**（1）DFA介绍**

<!--b是当前状态-->

<!--c是字符的java的hashcode-->

<!--p是转移后的状态-->

<!--value数组存储词语-->

<!--check和base双数组维护键的字典序-->

DFA：当状态b接受字符c转移到状态p时，双数组满足以下条件则转移成功，否则转移失败

`p = base [b] + c`

`check[p] = base[b]`

**（2）DAT的DFA修改**

为了避免\0的结束字符同文本字符混淆，将文本Hashcode加一即可，即

`p = base [b] + c + 1 `  

`check[p] = base[b]`

**（3）如何取到词语**

当状态p满足base[p]<0时，改状态对应单词结尾，其字典序为-base[p]-1

**（4）DFA构造**

![image-20221019204112305](https://cdn.jsdelivr.net/gh/Ga7777/Notes_NLPbook/imgs/202210211158018.png)

![image-20221019204145404](https://cdn.jsdelivr.net/gh/Ga7777/Notes_NLPbook/imgs/202210211158019.png)

![image-20221019204205412](https://cdn.jsdelivr.net/gh/Ga7777/Notes_NLPbook/imgs/202210211158020.png)

![image-20221019204217703](https://cdn.jsdelivr.net/gh/Ga7777/Notes_NLPbook/imgs/202210211158021.png)

![image-20221019204233386](https://cdn.jsdelivr.net/gh/Ga7777/Notes_NLPbook/imgs/202210211158022.png)

相关双数组构造过程可见知乎文章：[DoubleArrayTrie（DAT）双数组字典树原理解读，golang/go语言实现 - 知乎 (zhihu.com) ](https://zhuanlan.zhihu.com/p/113262718)

ps.其中offset就是我们要自己填的base数组。base数组的构建是需要启发式算法查找的（枚举也可以，比较慢）

- **性能分析：**虽然状态转移时间复杂度都是常数，但是由于每次挪动起点发起新的查询，全切分长度为n的文本时，最坏情况下复杂度为***O(n^2)***。e.g.假设字典中收录了所有数字，那么扫描123的状态转移情况为：1、12、123、2、23、3，一共发生了n+(n-1)+(n-2)+...+1次。

  双数组字典树能在O(n)时间复杂度内完成单串匹配（n是模式串长度），并且内存消耗可控。

## 2.3 AC自动机

<!--多模式匹配常用-->

- **提出动机**：DAT时间复杂度为***O(n^2)***。为使扫描的时间复杂度降为***O(n)***，提出AC自动机。

![image-20221019211121360](https://cdn.jsdelivr.net/gh/Ga7777/Notes_NLPbook/imgs/202210211158023.png)

- **基础知识：**由goto表（前缀树但首字带自环）、fail表（后缀树）、output表组成。

e.g. 模式串为{he,his,she,hers}

（1）goto表：

![image-20221019212703257](https://cdn.jsdelivr.net/gh/Ga7777/Notes_NLPbook/imgs/202210211158024.png)

（2）output表：

![image-20221019212732891](https://cdn.jsdelivr.net/gh/Ga7777/Notes_NLPbook/imgs/202210211158025.png)

（3）fail表

![image-20221019212755022](https://cdn.jsdelivr.net/gh/Ga7777/Notes_NLPbook/imgs/202210211158026.png)

![image-20221019212825257](https://cdn.jsdelivr.net/gh/Ga7777/Notes_NLPbook/imgs/202210211158027.png)

![image-20221019212853698](https://cdn.jsdelivr.net/gh/Ga7777/Notes_NLPbook/imgs/202210211158028.png)

对过程做一下详细介绍说明：

（1）与初始态相连的1和3的fail都指向初始态。

（2）

- **BFS第一轮：**

|   当前状态S    |        h1         |     s3     |
| :------------: | :---------------: | :--------: |
|     字符c      |        e/i        |     h      |
|   转移状态T    |        2/6        |     4      |
|  回溯F初始态   |     1.fail=0      |  3.fail=0  |
|   F.goto(c)    | 0(!null)/0(!null) |  1(!null)  |
| fail表指针情况 | 2.fail=0/6.fail=0 | 4.fail=1** |

- **BFS第二轮：**

|   当前状态S    |   he2    |   hi6    |   sh4    |
| :------------: | :------: | :------: | :------: |
|     字符c      |    r     |    s     |    e     |
|   转移状态T    |    8     |    7     |    5     |
|  回溯F初始态   | 2.fail=0 | 6.fail=0 | 4.fail=1 |
|   F.goto(c)    | 0(!null) | 3(!null) | 2(!null) |
| fail表指针情况 | 8.fail=0 | 7.fail=3 | 5.fail=2 |

- **BFS第三轮：**

|   当前状态S    |   her8   |
| :------------: | :------: |
|     字符c      |    s     |
|   转移状态T    |    9     |
|  回溯F初始态   | 8.fail=0 |
|   F.goto(c)    | 3(!null) |
| fail表指针情况 | 9.fail=3 |

- **性能分析：字典树转移状态失败时起点要向右挪一下重新扫描。而在AC自动集中，被goto表转移失败就按照fail表转移，永远不会失败，因此只需扫描一边文本。DAT时间复杂度为*O(n^2)*，AC自动机扫描的时间复杂度降为*O(n)*。且可以完成多模式匹配任务。**

## **2.4 ACDAT（基于DAT的AC自动机）**

- **提出动机：DAT速度比AC自动机速度更快。但AC自动机处理多模式匹配问题不需要回退，只要一遍扫描（处理多模式匹配问题更好）；而DAT处理但模式匹配问题复杂度低为O(n)，多模式匹配时需要频繁回退，性能较低。所以将两者结合起来。**
- **构建方式：**

**![image-20221019222439135](https://cdn.jsdelivr.net/gh/Ga7777/Notes_NLPbook/imgs/202210211158029.png)**

- **性能分析：ACDAT≈DAT>AC>BinTrie。ACDAT≈DAT是因为汉语词汇都不太长，有的甚至是单字，这使得前缀树的优势占了较大的比重，AC自动机的fail机制用处较小。所以含有短模式串时优先使用DAT，否则优先使用ACDAT。当最短词长大于2时ACDAT优势比较明显。**

## **2.5 准确率测评**

### **2.5.1 P/R/F1**

**![image-20221019223942798](https://cdn.jsdelivr.net/gh/Ga7777/Notes_NLPbook/imgs/202210211158030.png)**

**精确率P=TP/TP+FP**

**召回率R=TP/TP+FN**

**F1=2PR/(P+R)**

**以上针对分类问题，中文分词是分块问题。**

**每个文本对应的起始位置记作区间[i,j]，标准答案的所有区间构成集合A，为正类；分词结果所有单词区间构成集合B，那么**

**TP∪FN = A**

**TP∪FP = B**

**TP = A ∩ B**

**所以，P = |A∩B| / |B| ; R =| A∩B| / |A|**

**e.g.**

**![image-20221019230128136](https://cdn.jsdelivr.net/gh/Ga7777/Notes_NLPbook/imgs/202210211158031.png)**

### **2.5.2 中文分词测评**

**考虑对未登录词（新词）召回率，即OOV_Recall；登陆词召回率，IV_Recall。若对IV都无法百分百召回，说明词典分词的消歧能力不好。**

**<!--消除歧义在之后讲解。-->**

## **2.6 字典树其他应用**

**停用词过滤、简繁转化、拼音转换（字转拼音，涉及多音字，应该按词转换）。**

**调用Hanlp接口。**

# **第三章 二元语法与中文分词**

## **3.1 语言模型**

**给定一个句子w,语言模型计算句子出现概率p(w).**

**（1）原始语言模型**

**![image-20221020163359024](https://cdn.jsdelivr.net/gh/Ga7777/Notes_NLPbook/imgs/202210211158032.png)**

**![image-20221020163438206](https://cdn.jsdelivr.net/gh/Ga7777/Notes_NLPbook/imgs/202210211158033.png)**

**问题：**

**![image-20221020163458617](https://cdn.jsdelivr.net/gh/Ga7777/Notes_NLPbook/imgs/202210211158034.png)**

**（2）马尔科夫链与二元语法**

**![image-20221020163533634](https://cdn.jsdelivr.net/gh/Ga7777/Notes_NLPbook/imgs/202210211158035.png)**

**![image-20221020163552328](https://cdn.jsdelivr.net/gh/Ga7777/Notes_NLPbook/imgs/202210211158036.png)**

**（3）n元语法与RNN**

**![image-20221020163630551](https://cdn.jsdelivr.net/gh/Ga7777/Notes_NLPbook/imgs/202210211158037.png)**

**（4）数据稀疏与平滑策略**

- **数据稀疏：对于n元语法，n越大，数据稀疏现象越严重。**
- **平滑策略：最简单的是线性平滑，用λ来“劫富济贫”。**

## **3.2 模型训练**

1. **加载语料库**

```python
def load_cws_corpus(corpus_path):
    return CorpusLoader.convert2SentenceList(corpus_path)


if __name__ == '__main__':
    corpus_path = my_cws_corpus()
    sents = load_cws_corpus(corpus_path)
    for sent in sents:
        print(sent)
```

```python
('''商品 和 服务
商品 和服 物美价廉
服务 和 货币''')

结果：
[商品, 和, 服务]
[商品, 和服, 物美价廉]
[服务, 和, 货币]
```

2. **统计一元语法**

```python
def train_bigram(corpus_path, model_path):
    sents = CorpusLoader.convert2SentenceList(corpus_path)
    for sent in sents:
        for word in sent:
            if word.label is None:
                word.setLabel("n")
    maker = NatureDictionaryMaker()
    maker.compute(sents)
    maker.saveTxtTo(model_path)  # tests/data/my_cws_model.txt
```

**![image-20221020164742128](https://cdn.jsdelivr.net/gh/Ga7777/Notes_NLPbook/imgs/202210211158038.png)**

**<!--这里词性是没有用的-->**

3. **统计二元语法**

**![image-20221020165010882](https://cdn.jsdelivr.net/gh/Ga7777/Notes_NLPbook/imgs/202210211158039.png)**

## **3.3 模型预测**

1. **加载模型**

```python
def load_bigram(model_path, verbose=True, ret_viterbi=True):
    HanLP.Config.CoreDictionaryPath = model_path + ".txt"  # unigram
    HanLP.Config.BiGramDictionaryPath = model_path + ".ngram.txt"  # bigram
    CoreDictionary = SafeJClass('com.hankcs.hanlp.dictionary.CoreDictionary'):
    CoreBiGramTableDictionary = SafeJClass('com.hankcs.hanlp.dictionary.CoreBiGramTableDictionary'):
    print(CoreDictionary.getTermFrequency("商品")) # output 2
    print(CoreBiGramTableDictionary.getBiFrequency("商品", "和"))  # output 1
   
//getBiFrequency的参数为string类型，后续会先通过trie树转化为Id,在进行接下来的步骤时使用idA和idB来查询相关内容。
```

2. **底层优化实现**

**分析：CoreDictionary采用DAT存储词典，性能极高；但 CoreBiGramTableDictionary只能通过CoreDictionary查询词语id，再根据两个词语id去查询二元语法频次，如果直接用DAT存储二元词组，内存占用大且速度也不高。所以作者做了底层的优化——压缩算法。**

**只用到两个整型数组pair[]和start[]，目的是节省内存。**

**两个数组介绍：**

```java
/**
 * 描述了词在pair中的范围，具体说来
 * 给定一个词idA，从pair[start[idA]]开始的start[idA + 1] - start[idA]描述了一些接续的频次
 */
static int start[]; //特别注意！！一个idA对应两个idB时，offset是2，并不是4，不是对应的pair中的下标

/**
 * pair[偶数n]表示key，pair[n+1]表示frequency
 */
static int pair[];
```

**两个数组的构造过程：**

```java
public class CoreBiGramTableDictionary
{
    //先定义，再加载词典...
    //源码中有一句TreeMap<Integer, TreeMap<Integer, Integer>> map = new TreeMap<Integer, TreeMap<Integer, Integer>>();注意map的类型。
    
    //下边先把<idA,<idB,freq>>读取进来并按照这个格式存好，数组大小开辟好。一个idA对应多个<idB,freq>.
        try
        {
            br = new BufferedReader(new InputStreamReader(IOUtil.newInputStream(path), "UTF-8"));
            String line;
            int total = 0;
            int maxWordId = CoreDictionary.trie.size();
            while ((line = br.readLine()) != null)
            {
                String[] params = line.split("\\s");
                String[] twoWord = params[0].split("@", 2);
                String a = twoWord[0];
                int idA = CoreDictionary.trie.exactMatchSearch(a);
                if (idA == -1)
                {
                    continue;
                }
                String b = twoWord[1];
                int idB = CoreDictionary.trie.exactMatchSearch(b);
                if (idB == -1)
                {
                    continue;
                }
                int freq = Integer.parseInt(params[1]);
                TreeMap<Integer, Integer> biMap = map.get(idA); //注意这个类型，也就是说，对于每个idA词语，都有其对应的treemap存储对应的idB与A@B的词频。
                if (biMap == null)
                {
                    biMap = new TreeMap<Integer, Integer>();
                    map.put(idA, biMap);
                }
                biMap.put(idB, freq);
                total += 2; //total为pair数组大小，也就是说，每行读取完毕，total都要加2：一个key一个freq,最后就是所有二元组个数*2.
            }
            br.close();
            start = new int[maxWordId + 1]; //存储所有词语在pair数组中的开头对应
            pair = new int[total];  // total是接续的个数*2
            
            
            //下面对于每个词idA,取出它的<idB,freq>存入pair数组，并将idA对应pair中最开始的位置确定好，这样的话，就可以将pair中的<idB,freq>与相应的idA一一对应。
            int offset = 0;
            for (int i = 0; i < maxWordId; ++i)
            {
                TreeMap<Integer, Integer> bMap = map.get(i);
                if (bMap != null)
                {
                    for (Map.Entry<Integer, Integer> entry : bMap.entrySet())
                    {
                        int index = offset << 1;
                        pair[index] = entry.getKey();
                        pair[index + 1] = entry.getValue();
                        ++offset;
                    }
                }
                start[i + 1] = offset;
            }

        // 一些异常处理等操作，在此省略
        return true;
    }
```

**查询算法分为两步：**

```java
/**
 * 步骤1：利用二分查找（自己写）在pair数组的某个连续区间内搜索idB，得到下标index。该连续区间的起点是start[idA],搜索区间长度为start[idA+1]-start[idA].
 * 步骤2：c(A@B) = pair[index*2+1]
 */
int index = binarySearch(pair, start[idA], start[idA + 1] - start[idA], idB); 
//param分别为：目标数组、起始下标、区间长度、查询词的id
if (index < 0) return 0;
index <<= 1;
return pair[index + 1];
```

**这里的二分查找和普通的不一样，贴一下代码，后续例子中详细讲解：**

```java
private static int binarySearch(int[] a, int fromIndex, int length, int key)
{
    int low = fromIndex;
    int high = fromIndex + length - 1;

    while (low <= high)
    {
        int mid = (low + high) >>> 1; //>>>通过从左边推入零并让最右边的位脱落来右移,是无符号右移运算符,这里是为了获得只按照idB排列的下标（无frequency）
        int midVal = a[mid << 1]; //<<通过从右侧推入零并让最左边的位脱落来向左移动，这里是为了获得在pair中对应的下标

        if (midVal < key)
            low = mid + 1;
        else if (midVal > key)
            high = mid - 1;
        else
            return mid; // key found
    }
    return -(low + 1);  // key not found.
}
```

**e.g.  idA=trie("商品")=2，idB=trie("和")=0。map中存的是二元组的变种<idA,<idB,freq>>。map如下：**

**<0,<4,1>>**

**<0,<7,1>>**

**<1,<6,1>>**

**<2,<0,1>>**

**<2,<1,1>>**

**<3,<2,2>>**

**<3,<4,1>>**

**<4,<0,1>>**

**<4,<5,1>>**

**<6,<5,1>>**

**<7,<5,1>>**

**得到的pair[]和start[]如下：**

| **pair[]**  | 4,1,7,1 | 6,1  | 0,1,1,1 | 2,2,4,1 | 0,1,5,1 | null | 5,1  | 5,1  |      |
| :---------: | :-----: | :--: | :-----: | :-----: | :-----: | :--: | :--: | :--: | :--: |
| **start[]** |    0    |  2   |    3    |    5    |    7    |  7   |  9   |  10  |  11  |

```python
int index = binarySearch(pair, start[idA], start[idA + 1] - start[idA], idB); 
//在pair中找idB=0;  起始查找位置start[2]=3, 查找区间长度start[3]-start[2]=2
low=3,high=3+2-1=4
while(3<4):
    mid=(3+4)/2=3   //这里找的mid是只把idB排列起来的下标（因为start数组就是这样按照offset建立的，并不是对应的派人下标）
    midval=pair[6]==key(=0)
    return mid=3
    
return = pair[index*2+1] //先*2返回在pair中的key下标，再+1得到frequency
```

> **附txt文件内容：**
>
> **一元：**
>
> - **0和 n 2**
> - **1和服 n 1**
> - **2商品 n 2**
> - **3始##始 begin 3**
> - **4服务 n 2**
> - **5末##末 end 3**
> - **6物美价廉 n 1**
> - **7货币 n 1**
>
> **二元：**
>
> - **和@服务 1**
> - **和@货币 1**
> - **和服@物美价廉 1**
> - **商品@和  1**
> - **商品@和服 1**
> - **始##始@商品 2**
> - **始##始@服务 1**
> - **服务@和 1**
> - **服务@末##末 1**
> - **物美价廉@末##末 1**
> - **货币@末##末 1**

## **3.4 构建词网**

**词网是句子中所有一元语法组成的网状结构，起始位置相同的单词写作一行。**

**词网的创建是利用DAT的Seacher扫描出巨资中所有的医院词法及其位置而已，与词典分词的全切分概念类似。**

**词网构建代码如下：**

```python
def generate_wordnet(sent, trie):
    """
    生成词网
    :param sent: 句子
    :param trie: 词典（unigram）
    :return: 词网
    """
    searcher = trie.getSearcher(JString(sent), 0)
    wordnet = WordNet(sent)
    while searcher.next():
        wordnet.add(searcher.begin + 1,
                    Vertex(sent[searcher.begin:searcher.begin + searcher.length], searcher.value, searcher.index))
    # 原子分词，保证图连通。 要求：保证从起点出发的所有路径都会连通到终点
    vertexes = wordnet.getVertexes()
	/*vertexes:
	*0[ ]
	*1[商品]
	*2[] //品没有
	*3[和，和服]
	*4[服务]
	*5[] //务没有
	*6[ ]
	*/
    i = 0
    while i < len(vertexes):
        if len(vertexes[i]) == 0:  # 空白行
            j = i + 1
            for j in range(i + 1, len(vertexes) - 1):  # 寻找第一个非空行 j
                if len(vertexes[j]):
                    break
            oov = Vertex.newPunctuationInstance(sent[i - 1: j - 1])
            oov.attribute.totalFrequency = 1  # 将未登录词的词频设置为一个常数，比如1
            oov.wordID = 0
            wordnet.add(i, oov)  # 填充[i, j)之间的空白行
            i = j
        else:
            i += len(vertexes[i][-1].realWord) //realword是不带##的真实文本
    /*
    *i=0时，i=i+1
    *i=1时，i=i+2=3
    *i=3时，i=i+2=5 //为什么用最后一个词？策略罢了，并非能解决全部问题
    *i=5时j=6,sent[4,5]="务"，vertexes[5]="务",i=6
    *i=6时，i=6+1=7
    *跳出
    */
    	/*
        *wordnet：
        *0:[ ]
		*1:[商品]
		*2:[]
		*3:[和,和服]
		*4:[服务]
		*5:[务]
		*6:[ ]
    	*/
    return wordnet
```

**![image-20221020235338763](https://cdn.jsdelivr.net/gh/Ga7777/Notes_NLPbook/imgs/202210211158040.png)**

**![image-20221020235349174](https://cdn.jsdelivr.net/gh/Ga7777/Notes_NLPbook/imgs/202210211158041.png)**

**相关详细词网构建可参考：[HanLP分词研究 - 大熊猫同学 - 博客园 (cnblogs.com)](https://www.cnblogs.com/hapjin/p/11172299.html) （由于语料库的不同，我的单词被舍去，商、品、服等，但链接中留下了）**

## **3.5 节点距离计算**

**![image-20221020235626192](https://cdn.jsdelivr.net/gh/Ga7777/Notes_NLPbook/imgs/202210211158042.png)**

```java
    public static double calculateWeight(Vertex from, Vertex to)
    {
        int frequency = from.getAttribute().totalFrequency;
        if (frequency == 0)
        {
            frequency = 1;  // 防止发生除零错误
        }
//        int nTwoWordsFreq = BiGramDictionary.getBiFrequency(from.word, to.word);
        int nTwoWordsFreq = CoreBiGramTableDictionary.getBiFrequency(from.wordID, to.wordID);
        double value = -Math.log(dSmoothingPara * frequency / (MAX_FREQUENCY) + (1 - dSmoothingPara) * ((1 - dTemp) * nTwoWordsFreq / frequency + dTemp));
        if (value < 0.0)
        {
            value = -value;
        }
//        logger.info(String.format("%5s frequency:%6d, %s nTwoWordsFreq:%3d, weight:%.2f", from.word, frequency, from.word + "@" + to.word, nTwoWordsFreq, value));
        return value;
    }
```

**![image-20221021000204832](https://cdn.jsdelivr.net/gh/Ga7777/Notes_NLPbook/imgs/202210211158043.png)**

## **3.6 维特比算法**

### **3.6.1 词网转词图**

```java
public class Graph
{
    /**
     * 顶点
     */
    public Vertex[] vertexes;

    /**
     * 边，到达下标i
     */
    public List<EdgeFrom>[] edgesTo;

    /**
     * 将一个词网转为词图
     * @param vertexes 顶点数组
     */
    public Graph(Vertex[] vertexes)
    {
        int size = vertexes.length;
        this.vertexes = vertexes;
        edgesTo = new List[size];
        for (int i = 0; i < size; ++i)
        {
            edgesTo[i] = new LinkedList<EdgeFrom>();
        }
    }

    /**
     * 连接两个节点
     * @param from 起点
     * @param to 终点
     * @param weight 花费
     */
    public void connect(int from, int to, double weight)
    {
        edgesTo[to].add(new EdgeFrom(from, weight, vertexes[from].word + '@' + vertexes[to].word));
    }


    /**
     * 获取到达顶点to的边列表
     * @param to 到达顶点to
     * @return 到达顶点to的边列表
     */
    public List<EdgeFrom> getEdgeListTo(int to)
    {
        return edgesTo[to];
    }

    @Override
    public String toString()
    {
        return "Graph{" +
                "vertexes=" + Arrays.toString(vertexes) +
                ", edgesTo=" + Arrays.toString(edgesTo) +
                '}';
    }

    public String printByTo()
    {
        StringBuffer sb = new StringBuffer();
        sb.append("========按终点打印========\n");
        for (int to = 0; to < edgesTo.length; ++to)
        {
            List<EdgeFrom> edgeFromList = edgesTo[to];
            for (EdgeFrom edgeFrom : edgeFromList)
            {
                sb.append(String.format("to:%3d, from:%3d, weight:%05.2f, word:%s\n", to, edgeFrom.from, edgeFrom.weight, edgeFrom.name));
            }
        }

        return sb.toString();
    }

    /**
     * 根据节点下标数组解释出对应的路径
     * @param path
     * @return
     */
    public List<Vertex> parsePath(int[] path)
    {
        List<Vertex> vertexList = new LinkedList<Vertex>();
        for (int i : path)
        {
            vertexList.add(vertexes[i]);
        }

        return vertexList;
    }

    /**
     * 从一个路径中转换出空格隔开的结果
     * @param path
     * @return
     */
    public static String parseResult(List<Vertex> path)
    {
        if (path.size() < 2)
        {
            throw new RuntimeException("路径节点数小于2:" + path);
        }
        StringBuffer sb = new StringBuffer();

        for (int i = 1; i < path.size() - 1; ++i)
        {
            Vertex v = path.get(i);
            sb.append(v.getRealWord() + " ");
        }

        return sb.toString();
    }

    public Vertex[] getVertexes()
    {
        return vertexes;
    }

    public List<EdgeFrom>[] getEdgesTo()
    {
        return edgesTo;
    }
}
```

### **3.6.2 求词图上的最短路径问题**

- **暴力枚举：文本长度为n时，切分方式有2^(n-1)，复杂度是O(2^(n-1))，不可行**
- **动态规划：遍历过程中维护到某个节点是的最短路径，我们已经知道bellman-Ford算法和Dijkstra算法。**
- **维特比算法：从动态规划延伸到维特比，因为我们处理的是一种图的特例：由马尔可夫链构成的网状图，该特例上的最短路径算法称为维特比算法。**

***维特比算法分为前项和后向两个步骤：***

1. **前向：由起点出发从前往后遍历节点，更新从起点到该节点的最小花费以及前驱指针**
2. **后向：由终点出发从后往前回溯前驱指针，取得最短路径**

```python
def viterbi(wordnet):
    nodes = wordnet.getVertexes()
    # 前向遍历
    for i in range(0, len(nodes) - 1):
        for node in nodes[i]:
            for to in nodes[i + len(node.realWord)]:
                to.updateFrom(node)  # 根据距离公式计算节点距离，并维护最短路径上的前驱指针from
    # 后向回溯
    path = []  # 最短路径
    f = nodes[len(nodes) - 1].getFirst()  # 从终点回溯
    while f:
        path.insert(0, f)
        f = f.getFrom()  # 按前驱指针from回溯
    return [v.realWord for v in path]
```

**二元语法模型具备一定的泛化能力。以上的代码我们可以分好[商品，和，服务]，它已经在语料库中，那么我们尝试划分货品和服务，仍可以得到[货品，和，服务]。**

## **3.7 与用户字典的集成**

### **3.7.1 用户字典优先级**

**HanLP的所有分词器都支持用户字典，也就是说，允许用户自己设定一些词汇存在字典中。**

**且支持两档用户词典优先级：**

**![image-20221021111113574](https://cdn.jsdelivr.net/gh/Ga7777/Notes_NLPbook/imgs/202210211158045.png)**

**也就是说，低优先级下，分词器首先按照没有用户词典的情况下预测分词结果，再将结果按照用户词典合并。**

**高优先级下，先扫描用户词典将所有用户词语加入词网中（一元词网），此时二元语法频次依然缺失，为了参与统计模型运算，二元语法频次由程序伪造为与一元语法频次相同。这样用户可以通过词频进一步干预用户词语的优先级。（见3.7.3）**

### **3.7.2 模型测评**

**![image-20221021111929789](https://cdn.jsdelivr.net/gh/Ga7777/Notes_NLPbook/imgs/202210211158046.png)**

**![image-20221021112118478](https://cdn.jsdelivr.net/gh/Ga7777/Notes_NLPbook/imgs/202210211158047.png)**

### **3.7.3 模型调整**

**那么如何调整和干预？规律是什么？**

**词语的频次越大越容易切分，建议先选择一个较小的值，若不够再逐步增加。（词频越大，在这条路的花费越小，越容易按照这个词划分）**

- **3.7.2中前四个样本都可以用过用户词典来弥补。**

**![image-20221021112224384](https://cdn.jsdelivr.net/gh/Ga7777/Notes_NLPbook/imgs/202210211158048.png)**

- **3.7.2中第五个样本不可以通过一元语法词典解决。因为“输、气、管道”三个词语已经在核心词典里，只是没有被输出。追踪一下分词过程：**

**![image-20221021112439988](https://cdn.jsdelivr.net/gh/Ga7777/Notes_NLPbook/imgs/202210211158049.png)**



**![image-20221021113405451](https://cdn.jsdelivr.net/gh/Ga7777/Notes_NLPbook/imgs/202210211158050.png)**

**<!--模型调整的负面影响：人工新增二元语法的时候一元语法并没有得到更新，用户指定的频次也未必符合统计规律，可能产生一些副作用。总之除非万不得已，否则尽量用预料标注与统计方法解决问题。-->**

## **3.8 日语分词**

**用户无需关心语料下载，只要用train_bigram训练模型，然后load_bigram加载模型，然后使用该模型进行预测分词计科。并不需要精通日语，通过代码即可训练出一个可用的日语分词器。（即不是语言专家也可以通过机器学习设计出专业系统，只需要有语料库，NLP工程师就能教机器完成相应任务。）**



**然而，OOV召回依然是n元语法模型的硬伤，我们需要更强大的统计模型。**