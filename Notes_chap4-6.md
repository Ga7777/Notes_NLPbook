

# 第四章 隐马尔可夫模型与序列标注

<!--这张主要是为了解决：新词召回率低下的问题。-->

第三章的n元语法从**词语接续流畅度**出发，为**全切分词网**中的**二元接续**打分，进而利用维比特算法求解**似然概率最大的路径**。这种词语级别的模型无法应对OOV问题：OOV在最初全切分阶段就已经不可能进入词网了，更不能召回。（**词语级别的模型天然缺乏OOV召回能力**，我们需要**更细颗粒度**的模型——**字符级别**的模型）

只要将每个汉字组词所处的位置（首尾等）作为标签，中文分词就转化为给定汉字序列找出标签序列的问题。

由字构词是“序列标注”模型的一种应用。隐马尔科夫模型是其中最基础的一种。

## 4.1 序列标注问题

给定一个序列**x**=x1x2...xn，，找出每个元素对应标签**y**=y1y2...yn。求解序列标注问题的模型一般称为序列标注器，通常由模型从一个标注数据集{X,Y}={( x(i),y(i) )},i=1,...,K 中学习相关知识后再进行预测。

**序列标注与其他问题的结合**

- **中文分词**：转化为标注集为{切，过}的序列标注问题。流行的分词标注集：{B,M,E,S}

![image-20221029202832580](C:%5CUsers%5C%E4%B8%AD%E5%8D%97%E5%AE%89%E7%90%AA%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20221029202832580.png)

![image-20221029203004063](C:%5CUsers%5C%E4%B8%AD%E5%8D%97%E5%AE%89%E7%90%AA%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20221029203004063.png)

- **词性标注**：x是单词序列，y是对应的词性序列。词性标注集有很多不同的，颗粒度也不同。

<!--注意：词性标注需要综合考虑前后单词与磁性才能决定当前单词的词性。需要使用概率模型去模拟。-->

![image-20221029203023327](C:%5CUsers%5C%E4%B8%AD%E5%8D%97%E5%AE%89%E7%90%AA%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20221029203023327.png)

- **命名实体识别**：简短的人名和地名通过中文分词切分，然后通过词性标注确定所属列别。但有复合词时（丰度较小，导致分词器和词性标注器很难将其一步识别出来），需要在分词和词性标注的中间结果上进行召回。

<!--命名实体是OOV主要的组成部分-->

![image-20221029203139764](C:%5CUsers%5C%E4%B8%AD%E5%8D%97%E5%AE%89%E7%90%AA%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20221029203139764.png)

## 4.2 隐马尔可夫模型（HMM）

Hidden Markov Model

### 4.2.1 模型假设

HMM描述**两个时序序列联合分布p(x,y)的概率模型**，**x**序列外界可见（观测者），称为观测序列；**y**序列外界不可见，称为状态序列。

<!--它满足马尔科夫假设：每个事件的发生概率只取决于前一个事件。将满足该假设多个事件串联在一起就构成了马尔科夫链。在NLP特定语境下，事件具象成单词，马尔可夫模型具象为二元语法模型。-->

**隐马尔可夫模型两个假设：**

- 假设1：隐马尔可夫模型的马尔可夫假设作用于状态序列，当前状y_t仅仅依赖于前一个状态y_t-1，连续多个状态构成隐马尔可夫链**y**
- 假设2：任意时刻的观测状态x_t只依赖于该时刻的状态y_t，与其他时刻的状态或观测独立无关

![image-20221029204222836](C:%5CUsers%5C%E4%B8%AD%E5%8D%97%E5%AE%89%E7%90%AA%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20221029204222836.png)

贝叶斯定理可以解释上图的箭头：

![image-20221029204336494](C:%5CUsers%5C%E4%B8%AD%E5%8D%97%E5%AE%89%E7%90%AA%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20221029204336494.png)

### 4.2.2 模型三要素

1. 初始状态概率向量Π

![image-20221029204536063](C:%5CUsers%5C%E4%B8%AD%E5%8D%97%E5%AE%89%E7%90%AA%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20221029204536063.png)

2. 状态转移概率矩阵A

![image-20221029204608869](C:%5CUsers%5C%E4%B8%AD%E5%8D%97%E5%AE%89%E7%90%AA%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20221029204608869.png)

![image-20221029204654133](C:%5CUsers%5C%E4%B8%AD%E5%8D%97%E5%AE%89%E7%90%AA%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20221029204654133.png)

3. 发射概率矩阵B

![image-20221029204714657](C:%5CUsers%5C%E4%B8%AD%E5%8D%97%E5%AE%89%E7%90%AA%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20221029204714657.png)

![image-20221029204753008](C:%5CUsers%5C%E4%B8%AD%E5%8D%97%E5%AE%89%E7%90%AA%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20221029204753008.png)

三要素作用示意图如下：

![image-20221029204829956](C:%5CUsers%5C%E4%B8%AD%E5%8D%97%E5%AE%89%E7%90%AA%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20221029204829956.png)



***一个案例：医疗诊断***

![image-20221029205406976](C:%5CUsers%5C%E4%B8%AD%E5%8D%97%E5%AE%89%E7%90%AA%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20221029205406976.png)

![image-20221029205447388](C:%5CUsers%5C%E4%B8%AD%E5%8D%97%E5%AE%89%E7%90%AA%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20221029205447388.png)

```java
/**
 * 隐状态
 */
enum Status
{
    Healthy,
    Fever,
}

/**
 * 显状态
 */
enum Feel
{
    normal,
    cold,
    dizzy,
}
/**
 * 初始状态概率矩阵
 */
static float[] start_probability = new float[]{0.6f, 0.4f};
/**
 * 状态转移概率矩阵
 */
static float[][] transition_probability = new float[][]{
    {0.7f, 0.3f},
    {0.4f, 0.6f},
};
/**
 * 发射概率矩阵
 */
static float[][] emission_probability = new float[][]{
    {0.5f, 0.4f, 0.1f},
    {0.1f, 0.3f, 0.6f},
};
/**
 * 某个病人的观测序列
 */
static int[] observations = new int[]{normal.ordinal(), cold.ordinal(), dizzy.ordinal()};
```

### 4.2.3 三个基本用法

1. 样本生成问题：给定模型λ=（Π，A，B），生成满足模型约束的样本，即一系列观测序列x与对应的y
2. 模型训练问题：给定训练集(x,y)，估计模型参数λ=（Π，A，B）
3. 序列预测问题：已知模型参数λ=（Π，A，B）, 给定观测序列x，求最可能的状态序列y

## 4.3 样本生成

**样本生成算法：**

![image-20221029205542724](C:%5CUsers%5C%E4%B8%AD%E5%8D%97%E5%AE%89%E7%90%AA%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20221029205542724.png)

```java
/*
* xy为呀根本序列，采用二维数组存储。
* 约定xy[0][t]表示x_t,xy[1][t]表示y_t。
* drawFrom函数作用为：给定离散型随机变量的概率向量，采样一个变量值。
* 每个时刻t都先采样隐状态，然后采样下一个隐状态；
* 对每个隐状态，都仅仅根据发射概率矩阵B来采样显状态
*/

/**
 * 生成样本序列
 *
 * @param length 序列长度
 * @return 序列
 */
public abstract int[][] generate(int length){
    ...
    int xy[][] = new int[2][length];
    xy[1][0] = drawFrom(pi); //采样首个隐状态
    xy[0][0] = drawFrom(B[xy[1][0]]); //根据首个隐状态采样它的显状态
    for (int t = 1; t < length; t++)
    {
        xy[1][t] = frawFrom(A[xy[1][t-1]]);
        xy[0][t] = drawFrom(B[xy[1][t]]);
    }
    return xy;
}


    /**
     * 采样
     *
     * @param cdf 累积分布函数
     * @return
     */
    protected static int drawFrom(double[] cdf)
    {
        return -Arrays.binarySearch(cdf, Math.random()) - 1;
    }

```

|                                   |             0              |          1           | ...  |
| :-------------------------------: | :------------------------: | :------------------: | :--: |
|   (xt<cold/normal/dizzy>) xy[0]   | 2. 根据首隐和B来采样显状态 | 4. 根据y_1和B采样x_1 | ...  |
| (yt<healthy/fever>)         xy[1] |       1. 首个隐元素        | 3. 根据y_0和A采样y_1 | ...  |

**考虑到实际运用中往往要生成多个样本序列，指定一个size也可以生成多个序列：**

```java
/**
 * 生成样本序列
 *
 * @param minLength 序列最低长度
 * @param maxLength 序列最高长度
 * @param size      需要生成多少个
 * @return 样本序列集合
 */
public List<int[][]> generate(int minLength, int maxLength, int size)
{
    List<int[][]> samples = new ArrayList<int[][]>(size);
    for (int i = 0; i < size; i++)
    {
        samples.add(generate((int) (Math.floor(Math.random() * (maxLength - minLength)) + minLength)));
    }
    return samples;
}

// 测试生成2个[3,5]长度的的序列
for (int[][] sample : givenModel.generate(3, 5, 2))
{
     for (int t = 0; t < sample[0].length; t++)
     System.out.printf("%s/%s ", Feel.values()[sample[0][t]], Status.values()[sample[1][t]]);
     System.out.println();
}

/* 
* output:
* cold/Healthy cold/Healthy cold/Healthy 
* cold/Healthy normal/Fever normal/Healthy normal/Fever 
*/
```

由于随机数的原因，以上结果是随机的，但一定满足隐马尔科夫模型的约束。

**如何验证**：给定模型P，利用P生成的大量样本训练新模型Q，如果P和Q参数一致，则生成算法和训练算法极大概率都是正确的。

## 4.4 模型训练

*隐马尔可夫模型的**无监督学习可由EM算法**实现，称作Baum-Welch算法。详见李航《统计学习方法》第10章。*

隐马尔可夫模型的**监督学习利用极大似然法**来估计模型参数。<!--（频次求概率）-->

### 4.4.1 转移概率矩阵的估计

![image-20221029212513754](C:%5CUsers%5C%E4%B8%AD%E5%8D%97%E5%AE%89%E7%90%AA%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20221029212513754.png)

```java
/**
 * 利用极大似然估计转移概率
 *
 * @param samples   训练样本集
 * @param max_state 状态的最大下标，等于N-1
 */
protected void estimateTransitionProbability(Collection<int[][]> samples, int max_state)
{
    transition_probability = new float[max_state + 1][max_state + 1];
    for (int[][] sample : samples)
    {
        int prev_s = sample[1][0];
        for (int i = 1; i < sample[0].length; i++)
        {
            int s = sample[1][i];
            ++transition_probability[prev_s][s];
            prev_s = s;
        }
    }
    for (int i = 0; i < transition_probability.length; i++)
        normalize(transition_probability[i]);
}
```

### 4.4.2 初始状态概率向量的估计

![image-20221029212528822](C:%5CUsers%5C%E4%B8%AD%E5%8D%97%E5%AE%89%E7%90%AA%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20221029212528822.png)

```java
/**
 * 估计初始状态概率向量
 *
 * @param samples   训练样本集
 * @param max_state 状态的最大下标
 */
protected void estimateStartProbability(Collection<int[][]> samples, int max_state)
{
    start_probability = new float[max_state + 1];
    for (int[][] sample : samples)
    {
        int s = sample[1][0];
        ++start_probability[s];
    }
    normalize(start_probability);
}
```

### 4.4.3 发射概率矩阵的估计

![image-20221029212538425](C:%5CUsers%5C%E4%B8%AD%E5%8D%97%E5%AE%89%E7%90%AA%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20221029212538425.png)

```java
/**
 * 估计状态发射概率
 *
 * @param samples   训练样本集
 * @param max_state 状态的最大下标
 * @param max_obser 观测的最大下标
 */
protected void estimateEmissionProbability(Collection<int[][]> samples, int max_state, int max_obser)
{
    emission_probability = new float[max_state + 1][max_obser + 1];
    for (int[][] sample : samples)
    {
        for (int i = 0; i < sample[0].length; i++)
        {
            int o = sample[0][i];
            int s = sample[1][i];
            ++emission_probability[s][o];
        }
    }
    for (int i = 0; i < transition_probability.length; i++)
        normalize(emission_probability[i]);
}
```

### 4.4.4 完整的训练过程

```java
/**
 * 训练
 *
 * @param samples 数据集 int[i][j] i=0为观测，i=1为状态，j为时序轴
 */
public void train(Collection<int[][]> samples)
{
    if (samples.isEmpty()) return;
    int max_state = 0;
    int max_obser = 0;
    for (int[][] sample : samples)
    {
        if (sample.length != 2 || sample[0].length != sample[1].length) throw new IllegalArgumentException("非法样本");
        for (int o : sample[0])
            max_obser = Math.max(max_obser, o);
        for (int s : sample[1])
            max_state = Math.max(max_state, s);
    }
    estimateStartProbability(samples, max_state);
    estimateTransitionProbability(samples, max_state);
    estimateEmissionProbability(samples, max_state, max_obser);
    toLog(); //将概率参数取对数，用加法代替乘法，防止浮点数下溢出（多个小于0的数相乘趋于0）
}
```

### 4.4.5 验证：比较新旧模型参数

```java
public void testTrain() throws Exception
{
    FirstOrderHiddenMarkovModel givenModel = new FirstOrderHiddenMarkovModel(start_probability, transition_probability, emission_probability); //给定模型参数的模型
    FirstOrderHiddenMarkovModel trainedModel = new FirstOrderHiddenMarkovModel();
    trainedModel.train(givenModel.generate(3, 10, 100000)); //由给定模型生成样本训练而来的新模型
    assertTrue(trainedModel.similar(givenModel)); //用于比较两个模型参数误差是否在给定误差范围内
}
```

```java
   // 比较两个模型参数是否在给定误差范围内的函数
   public boolean similar(HiddenMarkovModel model)
    {
        if (!similar(start_probability, model.start_probability)) return false;
        for (int i = 0; i < transition_probability.length; i++)
        {
            if (!similar(transition_probability[i], model.transition_probability[i])) return false;
            if (!similar(emission_probability[i], model.emission_probability[i])) return false;
        }
        return true;
    }

    protected static boolean similar(float[] A, float[] B)
    {
        final float eta = 1e-2f;
        for (int i = 0; i < A.length; i++)
            if (Math.abs(A[i] - B[i]) > eta) return false;
        return true;
    }
}
```

## 4.5 模型预测

**隐马尔可夫最具有实际意义的问题：序列标注。**给定观测序列，求解最可能的状态序列及其概率。

### 4.5.1 概率计算的前向算法

![image-20221029213530051](C:%5CUsers%5C%E4%B8%AD%E5%8D%97%E5%AE%89%E7%90%AA%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20221029213530051.png)

### 4.5.2 搜索状态序列的维特比算法

1. 问题定义：

![image-20221029213959016](C:%5CUsers%5C%E4%B8%AD%E5%8D%97%E5%AE%89%E7%90%AA%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20221029213959016.png)

2. 解决方案

- ×暴力枚举：枚举每个时刻的N种备选状态(y)，相邻两个状态有N^2种组合，一共有T个时刻，复杂度O(TN^2)
- √动态规划（维特比）：递推，每次递推都在上一次的N挑局部路径中挑选，复杂度O(TN)。还要记录每个状态的前驱（用二维数组实现）。

3. 算法描述

![image-20221029214323628](C:%5CUsers%5C%E4%B8%AD%E5%8D%97%E5%AE%89%E7%90%AA%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20221029214323628.png)

4. 具体实现

```java
    public float predict(int[] observation, int[] state)
    {
        final int time = observation.length; // 序列长度
        final int max_s = start_probability.length; // 状态种数

        float[] score = new float[max_s];

        // link[t][s] := 第t个时刻在当前状态是s时，前1个状态是什么
        int[][] link = new int[time][max_s];
        // 第一个时刻，使用初始概率向量乘以发射概率矩阵
        for (int cur_s = 0; cur_s < max_s; ++cur_s)
        {
            score[cur_s] = start_probability[cur_s] + emission_probability[cur_s][observation[0]];
        }

        // 第二个时刻，使用前一个时刻的概率向量乘以一阶转移矩阵乘以发射概率矩阵
        float[] pre = new float[max_s];
        for (int t = 1; t < observation.length; t++)
        {
            // swap(now, pre)
            float[] _ = pre;
            pre = score;
            score = _;
            // end of swap
            // 状态转移仅仅利用了前一个时刻的状态y_t-1，所以算法不必记住那么久之前的分数（概率），只需记住前一个时刻的分数即可。所以上边用了滚动数组技巧，用pre数组存储前一个时刻的分数，与当前分数score交替滚动，节省了一些内存。滚动数组是动态规划时常用的技巧。
            for (int s = 0; s < max_s; ++s)
            {
                score[s] = Integer.MIN_VALUE;
                for (int f = 0; f < max_s; ++f)
                {
                    float p = pre[f] + transition_probability[f][s] + emission_probability[s][observation[t]];
                    if (p > score[s])
                    {
                        score[s] = p;
                        link[t][s] = f;
                    }
                }
            }
        }
        // link数组追踪路径，由于需要追踪每个时刻的最优路径，所以无法用滚动数组技巧
        
        // 两个数组推导到最终是课后，反向回溯，得到最佳路径及其概率
        float max_score = Integer.MIN_VALUE;
        int best_s = 0;
        for (int s = 0; s < max_s; s++)
        {
            if (score[s] > max_score)
            {
                max_score = score[s];
                best_s = s;
            }
        }
        
        for (int t = link.length - 1; t >= 0; --t)
        {
            state[t] = best_s;
            best_s = link[t][best_s];
        }

        return max_score;
    }
}
```

### 4.5.3 医疗诊断实例推算

- 问题：

在医疗诊断系统种，一位病人最近三天的身体感受是：正常、体寒、头晕，请预测他这三天最可能的健康状态和相应概率。

- 手算：

1. 根据下图得到Π，A，B参数

![img](file:///C:\Users\中南安琪\AppData\Roaming\Tencent\Users\827228087\QQ\WinTemp\RichOle\]D30JI63%]7}8AJQCC{4PUX.png)
$$
\pi =\{0.6,0.4\}
$$

$$
A =\{\{0.7,0.3\},\{0.4,0.6\}\}
$$

$$
B=\{\{0.5,0.4,0.1\},\{0.1,0.3,0.6\}\}
$$

***解释：***

Π：由初始状态转向第一个隐向量：

|  健康   |   发烧   |
| :-----: | :------: |
| pi1 0.6 | pai2 0.4 |

A：状态转移概率矩阵，从第二次开始状态转移的对应概率：

|                | 健康（到）A_,1 | 发烧（到）A_,2 |
| :------------: | :------------: | :------------: |
| 健康（从）A1,_ |      0.7       |      0.3       |
| 发烧（从）A2,_ |      0.4       |      0.6       |

B：发射概率矩阵：

|          | 正常B_,o1 | 体寒B_,o2 | 头晕B_,o3 |
| :------: | :-------: | :-------: | :-------: |
| 健康B1,_ |    0.5    |    0.4    |    0.1    |
| 发烧B2,_ |    0.1    |    0.3    |    0.6    |

2. 根据维比特算法更新score(\delta)和link(\psi)数组

已知N=2，M=3（预测三天，共两种状态<健康，发烧>）

（1）初始化
$$
\delta_1,_1=\pi_1B_1,o_1=0.6*0.5=0.3 转到健康还正常B_1,o_1\\
\psi_1,_1=0前驱是0
$$

$$
\delta_1,_2=\pi_2B_2,o_1=0.4*0.1=0.04 转到发烧还健康B_2,o_1 \\
\psi_1,_2=0前驱是0
$$

（2）递推

t=2时
$$
\delta_2,_1=max(\delta_1,_1A_1,_1,\delta_1,_2A_2,_1)B_1,o_2=max(0.3*0.7,0.04*0.4)*0.4=0.084 转到健康A_1,_1还体寒\\
\psi_2,_1=y_1前驱是健康\delta_1,_1
$$

$$
\delta_2,_2=max(\delta_1,_1A_1,_2,\delta_1,_2A_2,_2)B_2,o_2=max(0.3*0.3,0.04*0.6)*0.3=0.027 转到发烧A_1,_2还体寒\\
\psi_2,_2=y_1前驱是健康\delta_1,_1
$$

t=3时
$$
\delta_3,_1=max(\delta_2,_1A_1,_1,\delta_2,_2A_2,_1)B_1,o_3=max(0.084*0.7,0.027*0.4)*0.1=0.00588 转到健康A_1,_1还头晕\\
\psi_3,_1=y_1前驱是健康\delta_2,_1
$$

$$
\delta_3,_2=max(\delta_2,_1A_1,_2,\delta_2,_2A_2,_2)B_2,o_3=max(0.084*0.3,0.027*0.6)*0.6=0.01512 转到发烧A_1,_2还头晕A_1,_2\\
\psi_3,_2=y_1前驱是健康\delta_2,_1
$$

（3）终止 
$$
p^*=max(\delta_3,_1,\delta_3,_2)=0.01512 \\
i_3^*=argmax(\delta_3,_2)=y2 终点是发烧\delta_3,_2
$$
（4）回溯
$$
i_2^*=\psi_3,_2=健康y1(因为i_3^*=y2,对应下标2) \\
i_1^*=\psi_2,_1=健康y1(因为i_2^*=y1,对应下标1) \\
i_0^*=\psi_1,_1=0(因为i_1^*=y1,对应下标1)
$$
（5）整理验证

0——健康——健康——发烧

**验证**：
$$
0.6(健康)*0.5(健康还正常)*0.7(健康转健康)*0.4(健康还体寒)*0.3(健康转发烧)*0.6(发烧还头晕)=0.01512
$$

## 4.6 HMM模型用于中文分词

将标注集{B\M\E\S}映射为连续的整型id（为了提高运行时效率，整型比较快于字符串）

将字符映射为另一套连续id（字符作为观测变量，必须是整型才可被隐马尔科夫模型接受）。

1. 转换二元组

对《人民日报》格式的语料库，我们必须转换为(x,y)二元组才能训练模型。

```java
@Override
protected List<String[]> convertToSequence(Sentence sentence)
{
    List<String[]> charList = new LinkedList<String[]>();
    for (Word w : sentence.toSimpleWordList())
    {
        String word = CharTable.convert(w.value);
        if (word.length() == 1)
        {
            charList.add(new String[]{word, "S"}); //长度为1的识别为S：单个词
        }
        else
        {
            charList.add(new String[]{word.substring(0, 1), "B"}); //第一个字标为B
            for (int i = 1; i < word.length() - 1; ++i)
            {
                charList.add(new String[]{word.substring(i, i + 1), "M"}); //中间的标为M
            }
            charList.add(new String[]{word.substring(word.length() - 1), "E"}); //最后一个字标为E
        }
    }
    return charList;
}
```

2. 模型训练

```java
public void train(String corpus) throws IOException
{
    final List<List<String[]>> sequenceList = new LinkedList<List<String[]>>();
    IOUtility.loadInstance(corpus, new InstanceHandler() //将语料库的每个句子转换为字符串形式的二元组
    {
        @Override
        public boolean process(Sentence sentence)
        {
            sequenceList.add(convertToSequence(sentence));
            return false;
        }
    });

    TagSet tagSet = getTagSet();

    List<int[][]> sampleList = new ArrayList<int[][]>(sequenceList.size());
    for (List<String[]> sequence : sequenceList) //将字符串映射为相应的id
    {
        int[][] sample = new int[2][sequence.size()];
        int i = 0;
        for (String[] os : sequence)
        {
            sample[0][i] = vocabulary.idOf(os[0]);
            assert sample[0][i] != -1;
            sample[1][i] = tagSet.add(os[1]);
            assert sample[1][i] != -1;
            ++i;
        }
        sampleList.add(sample);
    }

    model.train(sampleList); //执行训练
    vocabulary.mutable = false; //训练完毕后将词表置为只读
}
```

```java
/*
 * 演示一阶隐马尔可夫模型用于序列标注问题之中文分词
 */

    private static Segment trainHMM(HiddenMarkovModel model) throws IOException
    {
        HMMSegmenter segmenter = new HMMSegmenter(model);//要求用户传入一个空白的HiddenMarkovModel以创建分词器
        segmenter.train(MSR.TRAIN_PATH); //给出路径，训练模型
        return segmenter.toSegment(); //将分词器转换为Segment接口，以兼容测评方法
    }

    Segment hmm = trainHMM(new FirstOrderHiddenMarkovModel()); //传入一阶隐马尔可夫模型
```

对应的Python代码：

```python
def train(corpus, model):
    segmenter = HMMSegmenter(model)
    segmenter.train(corpus)
    print(segmenter.segment('商品和服务'))
    return segmenter.toSegment()

if __name__ == '__main__':
    segment = train(msr_train, FirstOrderHiddenMarkovModel()
    
/*
* Output:
* [商品, 和, 服务]
*/
```

3. 模型预测

训练完模型后，模型的预测结果是{B,M,E,S}标签序列，分词器要根据标签序列指示，将字符序列转为单词序列。

![image-20221030154904934](C:%5CUsers%5C%E4%B8%AD%E5%8D%97%E5%AE%89%E7%90%AA%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20221030154904934.png)

```java
@Override
public void segment(String text, String normalized, List<String> output)
{
    int[] obsArray = new int[text.length()];
    for (int i = 0; i < obsArray.length; i++)
    {
        obsArray[i] = vocabulary.idOf(normalized.substring(i, i + 1));
    }
    int[] tagArray = new int[text.length()];
    model.predict(obsArray, tagArray);
    StringBuilder result = new StringBuilder();
    result.append(text.charAt(0));

    for (int i = 1; i < tagArray.length; i++)
    {
        if (tagArray[i] == tagSet.B || tagArray[i] == tagSet.S)
        {
            output.add(result.toString());
            result.setLength(0);
        }
        result.append(text.charAt(i));
    }
    if (result.length() != 0)
    {
        output.add(result.toString());
    }
}
```

e.g. 商品和服务

|      | 0    | 1    | 2    | 3    | 4    |
| ---- | ---- | ---- | ---- | ---- | ---- |
| x    | 商   | 品   | 和   | 服   | 务   |
| y    | B    | E    | S    | B    | E    |

i=0：W=[商] 

开始进入循环

i=1：target(1)=E，直接W+[品]=[商品]

i=2：target(2)=S，进入If，L+W=[“商品”]，W=[]，W+[和]=[和]

i=3：target(3)=B，进入If，L+W=[“商品”，“和”]，W=[]，W+[服]=[服]

i=4：target(4)=S，直接W+[务]=[服务]

最后W!=[]，L+W=[“商品”，“和”，“服务”]

4. 模型测评

```java
    public static void trainAndEvaluate(HiddenMarkovModel model) throws IOException
    { //标准化测评
        Segment hmm = trainHMM(model);
        CWSEvaluator.Result result = CWSEvaluator.evaluate(hmm, MSR.TEST_PATH, MSR.OUTPUT_PATH, MSR.GOLD_PATH, MSR.TRAIN_WORDS); //模型，原始句，模型预测输出，标准答案，词典
        System.out.println(result);
    }
```

```python
def evaluate(segment):
    result = CWSEvaluator.evaluate(segment, msr_test, msr_output, msr_gold, msr_dict)
    print(result)

/* 
msr_test = os.path.join(sighan05, 'testing', 'msr_test.utf8') //需要切分的
扬帆远东做与中国合作的先行
希腊的经济结构较特殊。
海运业雄踞全球之首，按吨位计占世界总数的１７％。
另外旅游、侨汇也是经济收入的重要组成部分，制造业规模相对较小。
多年来，中希贸易始终处于较低的水平，希腊几乎没有在中国投资。
十几年来，改革开放的中国经济高速发展，远东在崛起。
他感受到了中国经济发展的大潮。
他要与中国人合作。
msr_output = os.path.join(sighan05, 'testing', 'msr_bigram_output.txt') //hmm分词结果
扬帆  远东  做  与  中国  合作  的  先行
希腊  的  经济  结构  较  特殊  。
海运业  雄踞  全球  之首  ，  按吨  位计  占  世界  总数  的  １７％  。
另外  旅游  、  侨汇  也  是  经济  收入  的  重要  组成  部分  ，  制造业  规模  相对  较小  。
多年来  ，  中  希贸易始  终处  于  较低  的  水平  ，  希腊  几乎  没有  在  中国  投资  。
十几年来  ，  改革  开放  的  中国  经济  高速  发展  ，  远东  在  崛起  。
他  感受  到  了  中国  经济  发展  的  大潮  。
他要  与  中国  人  合作  。
msr_gold = os.path.join(sighan05, 'gold', 'msr_test_gold.utf8') //标准答案
扬帆  远东  做  与  中国  合作  的  先行  
希腊  的  经济  结构  较  特殊  。
海运  业  雄踞  全球  之  首  ，  按  吨位  计  占  世界  总数  的  １７％  。
另外  旅游  、  侨汇  也是  经济  收入  的  重要  组成部分  ，  制造业  规模  相对  较小  。
多年来  ，  中  希  贸易  始终  处于  较低  的  水平  ，  希腊  几乎  没有  在  中国  投资  。
十几年  来  ，  改革开放  的  中国  经济  高速  发展  ，  远东  在  崛起  。
他  感受  到  了  中国  经济  发展  的  大潮  。
他  要  与  中国人  合作  。
msr_dict = os.path.join(sighan05, 'gold', 'msr_training_words.utf8') //词典
１１２３项
义演
佳酿
沿街
老理
三四十岁
解波
统建
蓓蕾
李佑生
肾结石
劳作
海因兹
上海行政开发培训班
大声
２０余条
...
* 
*/


if __name__ == '__main__':
    segment = train(msr_train, FirstOrderHiddenMarkovModel())
    evaluate(segment)
    segment = train(msr_train, SecondOrderHiddenMarkovModel())
    evaluate(segment)
    
/*
* Output:
* [商品, 和, 服务]
* P:78.49 R:80.38 F1:79.42 OOV-R:41.11 IV-R:81.44
* [商品, 和, 服务]
* P:78.34 R:80.01 F1:79.16 OOV-R:42.06 IV-R:81.04
*/
```

## 4.7 二阶隐马尔可夫模型

- **提出动机**：从以上标准化测评结果可以看出：一阶HMM模型OOV召回率明显提高了，但是F1和IV召回率比二元模型降低了太多。说明模型太简单、欠拟合、IV记不住。所以尝试增加模型复杂度的方法。

- **改进方式**：

  （1）参数估计：Π和B与一阶HMM完全一致。A也用极大似然法估计，但是维度从二维变为三维：当t>=3时，将序列语法片段的频次按照下表(i,j,k)计入张量A(i,j,k)，归一化后得到相应概率。

  （2）解码时的维比特算法：必须考虑前两个状态。递推时，双重循环遍历所有可能的(si,sj)，维护两个数组。
  $$
  \delta_t,_i,_j=max_{1<=k<=N}(\delta_{t-2,_k,_iA_k,_i,_j})B_j,ot<i,j=1,...,N> \\
  \psi_t,_i,_j=argmax_{1<=k<=N}(\delta_{t-2,_k,_iA_k,_i,_j})<i,j=1,...,N> \\
  其中\delta_t,_i,_j表示时刻 t 时，以(s_i,s_j)结尾的路径的最大概率，\psi_t,_i,_j表示y_{t-2}
  $$
  *具体实现时要将score和link数组增加一维。且第1、2个时刻要特殊处理，第3时刻开始按公式循环递推。*

- **用于中文分词的测评**：

  和一阶一样，传入SecondOrderHiddenMarkovModel即可。

![image-20221030161211719](C:%5CUsers%5C%E4%B8%AD%E5%8D%97%E5%AE%89%E7%90%AA%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20221030161211719.png)

可以看到：隐马尔科夫模型召回了将近一半的OOV，但综合F1甚至低于词典分词。朴素的隐马尔科夫模型并不适合中文分词（仅仅提高HMM模型复杂度不可取），我们需要更高级的模型。

# 第五章 感知机分类与序列标注

<!--为了解决隐马尔科夫模型利用特征太少，导致准确率太低的问题。（同时进一步召回了OOV）-->

- 隐马尔可夫能捕捉到的特征只有两种：

1. 前一个标签是什么
2. 当前字符是什么

- 为了**利用更多特征**，我们应用线性模型，它由两部分构成：

1. 一系列用来提取特征的特征函数**φ**
2. 相应的权重向量**w**

## 5.1 线性分类模型与感知机算法

### 5.1.1 分类问题

我们知道，二分类问题可以推广到多分类（通过one-vs-one或者one-vs-rest）的方式。

1. one-vs-one：每次区别两类Ci与Cj，一共进行k(k-1)/2次，理想情况下有且仅有一种类别Ck每次都胜出，预测结果就为k类。但成本高。
2. ove-vs-rest：每次区别Ci和非Ci，一共进行k次二分类。理想情况下模型给予Ck的分数是所有K次分类中的最高值，预测结果就为k类。虽然成本低，但正负样本数量不均匀，会降低分类准确率。

### 5.1.2 线性分类模型

1. **组成部分**：由特征函数**φ**权重向量**w**组成。

```java
public class LinearModel implements ICacheAble
{
    /**
     * 特征函数
     */
    public FeatureMap featureMap; // featureMap将字符串形式的特征映射为独一无二的特征id，记作i。在接下来的例子中，featureMap.length = 1695922
    /**
     * 特征权重
     */
    public float[] parameter; // parameter存储着每个特征的权重wi。在接下来的例子中， parameter.length = 6783688


    public LinearModel(FeatureMap featureMap, float[] parameter)
    {
        this.featureMap = featureMap;
        this.parameter = parameter;
    }

    public LinearModel(FeatureMap featureMap)
    {
        this.featureMap = featureMap;
        parameter = new float[featureMap.size() * featureMap.tagSet.size()]; //6783688=1695922*4 不同的特征向量映射到不同的{B、M、E、S}的权重参数
    }
```

2. **特征向量与样本空间**

   ```java
   /**
    * 特征提取
    *
    * @param text       文本
    * @param featureMap 特征映射
    * @return 特征向量
    */
   protected abstract List<Integer> extractFeature(String text, FeatureMap featureMap);
   ```

通过FeatureMap（特征函数）来提取特征。在线性模型中，一般输出1或者0来表示样本是否含有该特征。由于特征向量中每个元素xi∈{0,1},且1的数量更少，所以只需要记录xi=1的下标i即可。这样省内存且计算量更小。

e.g. 姓名判性别问题

feature1：name.contains("雁")?1:0; 

feature2：name.contains("冰")?1:0; 

所以沈雁冰这个样本表示为向量x(1)=[1,1]，冰心表示为[0,1]

样本分布的空间称作样本空间。将数据转化为特征向量后，**分类问题就转化为样本空间的分割问题。**

3. **决策边界与分离超平面**

![image-20221101155220680](C:%5CUsers%5C%E4%B8%AD%E5%8D%97%E5%AE%89%E7%90%AA%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20221101155220680.png)

```java
/**
 * 分离超平面解码
 *
 * @param x 特征向量
 * @return sign(wx)
 */
public int decode(Collection<Integer> x)
{
    float y = 0;
    for (Integer f : x)
        y += parameter[f];
    return y < 0 ? -1 : 1;
}
```

***如果线性不可分，仍可以用线性模型——***

- 定义更多特征
- 使用特殊的函数将数据映射到高维空间去（参考SVM核函数）

### 5.1.3 感知机算法

1. **朴素感知机算法**

![image-20221101155507599](C:%5CUsers%5C%E4%B8%AD%E5%8D%97%E5%AE%89%E7%90%AA%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20221101155507599.png)

```java
    /**
     * 朴素感知机训练算法
     *  @param instanceList 训练实例
     * @param featureMap   特征函数
     * @param maxIteration 训练迭代次数
     */
    private static LinearModel trainNaivePerceptron(Instance[] instanceList, FeatureMap featureMap, int maxIteration)
    {
        LinearModel model = new LinearModel(featureMap, new float[featureMap.size()]);
        for (int it = 0; it < maxIteration; ++it)
        {
            Utility.shuffleArray(instanceList);
            for (Instance instance : instanceList)
            {
                int y = model.decode(instance.x);
                if (y != instance.y) // 误差反馈
                    model.update(instance.x, instance.y);
            }
        }
        return model;
    }

/**
 * 参数更新
 *
 * @param x 特征向量
 * @param y 正确答案
 */
public void update(Collection<Integer> x, int y)
{
    assert y == 1 || y == -1 : "感知机的标签y必须是±1";
    for (Integer f : x)
        parameter[f] += y;
}
```

***从直观角度和仿生学角度理解参数更新方式：***

![image-20221101155859479](C:%5CUsers%5C%E4%B8%AD%E5%8D%97%E5%AE%89%E7%90%AA%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20221101155859479.png)

***从梯度下降理解参数更新方式：***

![image-20221101160248844](C:%5CUsers%5C%E4%B8%AD%E5%8D%97%E5%AE%89%E7%90%AA%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20221101160248844.png)

- **存在的问题**

加入数据本身线性不可分的话，**感知机算法不收敛**，每次迭代分离超平面都会剧烈震荡。因为感知机是一个在线学习模型，学一个训练实例后就可以更新整个模型。1个噪声点就可以使前面完美的9999个正确分类点前功尽弃。

- **解决方案**

（1）创造更多特征，将样本映射到高维空间，使其线性可分

（2）切换到其他训练算法，如支持向量机

（3）对感知机算法打补丁，使用投票感知机和平均感知机

2. **投票感知机和平均感知机**

<!--是为了解决数据线性不可分的问题，平均感知机是收敛的-->

![image-20221101161053828](C:%5CUsers%5C%E4%B8%AD%E5%8D%97%E5%AE%89%E7%90%AA%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20221101161053828.png)

![image-20221101161159347](C:%5CUsers%5C%E4%B8%AD%E5%8D%97%E5%AE%89%E7%90%AA%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20221101161159347.png)

![image-20221101161240119](C:%5CUsers%5C%E4%B8%AD%E5%8D%97%E5%AE%89%E7%90%AA%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20221101161240119.png)

```java
/**
 * 平均感知机训练算法
 *  @param instanceList 训练实例
 * @param featureMap   特征函数
 * @param maxIteration 训练迭代次数
 */
private static LinearModel trainAveragedPerceptron(Instance[] instanceList, FeatureMap featureMap, int maxIteration)
{
    float[] parameter = new float[featureMap.size()];
    double[] sum = new double[featureMap.size()];
    int[] time = new int[featureMap.size()];

    AveragedPerceptron model = new AveragedPerceptron(featureMap, parameter);
    int t = 0;
    for (int it = 0; it < maxIteration; ++it)
    {
        Utility.shuffleArray(instanceList);
        for (Instance instance : instanceList)
        {
            ++t;
            int y = model.decode(instance.x);
            if (y != instance.y) // 误差反馈
                model.update(instance.x, instance.y, sum, time, t); //方法见下
        }
    }
    model.average(sum, time, t);//方法见下
    return model;
}
```

```java
/**
 * 根据答案和预测更新参数
 *
 * @param index     特征向量的下标
 * @param value     更新量
 * @param total     权值向量总和
 * @param timestamp 每个权值上次更新的时间戳
 * @param current   当前时间戳
 */
private void update(int index, float value, double[] total, int[] timestamp, int current)
{
    int passed = current - timestamp[index];
    total[index] += passed * parameter[index];
    parameter[index] += value;
    timestamp[index] = current;
}
```

```java
public void average(double[] total, int[] timestamp, int current)
{
    for (int i = 0; i < parameter.length; i++)
    {
        parameter[i] = (float) ((total[i] + (current - timestamp[i]) * parameter[i]) / current); //注意最后一次还要加！！
    }
}
```

e.g. 设d=3，为每个参数w1,w2,w3初始化累积量sum1=sum2=sum3=0,time1=time2=time3=0,t=0

|      |  w1  |  w2  |  w3  |
| :--: | :--: | :--: | :--: |
| t=0  |  0   |  0   |  0   |
| t=1  | 0.6  | 0.1  | 0.7  |
| t=2  | 0.6  | 0.1  | 0.5  |
| t=3  | 0.6  | 0.1  | 0.5  |
| t=4  | 0.5  | 0.2  | 0.6  |

- t=1，读入训练样本x(i),y(i)并执行预测，发现y!=y(i)，开始更新

sum1=sum1+(t-time1)\*w1=0+(1-0)\*0

sum2=sum2+(t-time2)\*w2=0+(1-0)\*0

sum3=sum3+(t-time3)\*w1=0+(1-0)\*0

time1=time2=time3=t=1

w1=0.6,w2=0.1,w3=0.7

- t=2，读入训练样本x(i),y(i)并执行预测，发现y!=y(i)，开始更新

sum3=sum3+(t-time3)\*w3=0+(2-1)*0.7=0.7

time3=t=2,

w3=0.5

- t=3，读入训练样本x(i),y(i)并执行预测，发现y=y(i)，不更新
- t=4，读入训练样本x(i),y(i)并执行预测，发现y!=y(i)，开始更新

sum1=sum1+(t-time1)\*w1=0+(4-1)\*0.6=3*0.6=1.8

sum2=sum2+(t-time2)\*w2=0+(4-1)\*0.1=3*0.1=0.3

sum3=sum3+(t-time3)\*w1=0.7+(4-2)\*0.5=0.7+2*0.5=1.7

time1=time2=time3=t=4

w1=0.5,w2=0.2,w3=0.6

再次重复读入实例进行上述操作，直到训练次数结束。

假如迭代次数结束，进入average，最后一次更新完的参数w还没计入，所以还需要再加 (current - timestamp[i]) * parameter[i]). 总量再除以当前的t

w1=(sum1+(t-time1)*w1)/t

<!--个人认为作者这里的算法有误，如果是最后一次迭代，最后一个样例读入后参数又更新了，那么当前时间t和最新修改的时间戳time应该是同一个数，相减的话无法加上最后这次的参数。应该t+1后执行average()-->

### 5.1.4 基于感知机的人名性别分类

![image-20221101172550105](C:%5CUsers%5C%E4%B8%AD%E5%8D%97%E5%AE%89%E7%90%AA%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20221101172550105.png)

（详见P174-178）

- 特征工程，可以将特征变得更复杂或者更简单，其对应的w参数也会变化。

![image-20221101172800394](C:%5CUsers%5C%E4%B8%AD%E5%8D%97%E5%AE%89%E7%90%AA%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20221101172800394.png)

## 5.2 结构化预测问题

**考虑结构整体的合理程度。**对于线性模型，训练算法为结构化感知机。

### 5.2.1 结构化感知机算法

![image-20221101193131931](C:%5CUsers%5C%E4%B8%AD%E5%8D%97%E5%AE%89%E7%90%AA%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20221101193131931.png)

**结构化感知机算法**：

（1）
$$
读入样本(x^{(i)},y^{(i)})，执行结构化预测\widehat{y}=argmax_{y∈Y}(w\cdot\phi(x^{(i)},y))
$$
（2）
$$
与正确答案对比，若\widehat{y}\neq y^{(i)},则更新参数：\\
奖励正确答案触发的特征函数的权重w \leftarrow w+\phi(x^{(i)},y^{(i)})\\
惩罚错误答案触发的特征函数的权重w \leftarrow w-\phi(x^{(i)},\widehat{y})\\
奖惩可以合并到一个式子里：w \leftarrow w+\phi(x^{(i)},y^{(i)})-\phi(x^{(i)},\widehat{y})\\
还可以调整学习率：w \leftarrow w+\alpha(\phi(x^{(i)},y^{(i)})-\phi(x^{(i)},\widehat{y}))
$$
相较于感知机算法，它主要不同在：

（1）修改了特征向量

（2）参数更新赏罚分明

```java
/**
 * 根据答案和预测更新参数
 *
 * @param goldIndex    答案的特征函数（非压缩形式）
 * @param predictIndex 预测的特征函数（非压缩形式）
 */
public void update(int[] goldIndex, int[] predictIndex)
{
    for (int i = 0; i < goldIndex.length; ++i)
    {
        if (goldIndex[i] == predictIndex[i])
            continue;
        else // 预测与答案不一致
        {
            parameter[goldIndex[i]]++; // 奖励正确的特征函数（将它的权值加一）
            if (predictIndex[i] >= 0 && predictIndex[i] < parameter.length)
                parameter[predictIndex[i]]--; // 惩罚招致错误的特征函数（将它的权值减一）
            else
            {
                throw new IllegalArgumentException("更新参数时传入了非法的下标");
            }
        }
    }
}
```

### 5.3.2 结构化感知机与序列标注

序列标注最大的结构特点是**：标签相互之间的依赖性。**

隐马尔科夫模型用：初始状态概率向量Π和状态转移概率矩阵A来体现。

线性模型中，利用特征来描述这种依赖性：

![image-20221101194849196](C:%5CUsers%5C%E4%B8%AD%E5%8D%97%E5%AE%89%E7%90%AA%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20221101194849196.png)

![image-20221101195015885](C:%5CUsers%5C%E4%B8%AD%E5%8D%97%E5%AE%89%E7%90%AA%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20221101195015885.png)

![image-20221101195040620](C:%5CUsers%5C%E4%B8%AD%E5%8D%97%E5%AE%89%E7%90%AA%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20221101195040620.png)

```java
/**
 * 通过命中的特征函数计算得分
 *
 * @param featureVector 压缩形式的特征id构成的特征向量
 * @return
 */
public double score(int[] featureVector, int currentTag) // 前一个状态转移到现在状态+当前这个字符的特征向量发射为当前状态的  得分
{
    double score = 0;
    System.out.println();
    System.out.println("开始计算score:");
    for (int index : featureVector) // featureVector.length = 8 存储了 7特征模板提取的特征向量 + 1转移向量
    {
        System.out.print(index+",");
        if (index == -1)
        {
            continue;
        }
        else if (index < -1 || index >= featureMap.size())
        {
            throw new IllegalArgumentException("在打分时传入了非法的下标");
        }
        else
        {
            index = index * featureMap.tagSet.size() + currentTag; // featureMap.tagSet.size() = 4，前边的操作我们可以看到，featureMap存储了1695922条不同特征映射Id，parameter是它的4倍长，即每个特征映射为B\M\E\S的特征权重参数，所以当前特征id——index对应的权重参数要先*4，再找到对应tag的权重参数。(为了对应到特征index对应的w)
            score += parameter[index];    // 其实就是特征权重的累加
            // parameter.length = 6783688
        }
    }
    System.out.println();
    System.out.println("这次计算score结束");
    return score;
}

//前7个是特征模板5-2表格提取的关于该字符的7个特征组成的特征向量，第8个是转移到的状态(si)特征。
//8维的特征向量有自己对应的w特征权重，累加这些特征权重就是我们要求的score

Output：
    
开始计算score:
138,35846,122824,22498,122825,122826,538222,3,
这次计算score结束

开始计算score:
138,35846,122824,22498,122825,122826,538222,0,
这次计算score结束

开始计算score:
138,35846,122824,22498,122825,122826,538222,1,
这次计算score结束

开始计算score:
138,35846,122824,22498,122825,122826,538222,2,
这次计算score结束

开始计算score:
138,35846,122824,22498,122825,122826,538222,3,
这次计算score结束

开始计算score:
138,35846,122824,22498,122825,122826,538222,0,
这次计算score结束

开始计算score:
138,35846,122824,22498,122825,122826,538222,1,
这次计算score结束

开始计算score:
138,35846,122824,22498,122825,122826,538222,2,
这次计算score结束

开始计算score:
138,35846,122824,22498,122825,122826,538222,3,
这次计算score结束

开始计算score:
35851,122828,500,122829,122830,538224,71302,0,
这次计算score结束
50745,2457,7855,841344,1010757,85717,
开始计算score:
1196,2992,49,2993,2994,2995,5208,2,
这次计算score结束

开始计算score:
1196,2992,49,2993,2994,2995,5208,3,
这次计算score结束

开始计算score:
2997,55,518,2998,2999,5210,161612,0,
这次计算score结束

开始计算score:
2997,55,518,2998,2999,5210,161612,1,
这次计算score结束

...
```

### 5.3.3 结构感知机的维特比解码算法

![image-20221101195448015](C:%5CUsers%5C%E4%B8%AD%E5%8D%97%E5%AE%89%E7%90%AA%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20221101195448015.png)

*（比对隐马尔可夫模型）*

```java
public double viterbiDecode(Instance instance)
{
    return viterbiDecode(instance, instance.tagArray);
}

/**
 * 维特比解码
 *
 * @param instance   实例
 * @param guessLabel 输出标签
 * @return
 */
public double viterbiDecode(Instance instance, int[] guessLabel)
{
    final int[] allLabel = featureMap.allLabels(); // featureMap.size() = 1695922 存储了这么多的特征向量映射id
    final int bos = featureMap.bosTag();
    final int sentenceLength = instance.tagArray.length;
    final int labelSize = allLabel.length;

    int[][] preMatrix = new int[sentenceLength][labelSize];
    double[][] scoreMatrix = new double[2][labelSize];

    for (int i = 0; i < sentenceLength; i++)
    {
        int _i = i & 1;
        int _i_1 = 1 - _i;
        int[] allFeature = instance.getFeatureAt(i); // allFeature.length = 8 ，存储了当前字符的8维特征向量（7特征工程提取的特征+1转移特征）
        final int transitionFeatureIndex = allFeature.length - 1; // transitionFeatureIndex = 7
        if (0 == i) //第一个做特殊处理，是从BOS开始的
        {
            allFeature[transitionFeatureIndex] = bos; //数组最后一个位置用来存储转移特征，5-2特征模板的7个特征+1个转移特征(共8个)
            for (int j = 0; j < allLabel.length; j++)
            {
                preMatrix[0][j] = j;

                double score = score(allFeature, j);

                scoreMatrix[0][j] = score;
            }
        }
        else
        {
            for (int curLabel = 0; curLabel < allLabel.length; curLabel++)
            {

                double maxScore = Integer.MIN_VALUE;

                for (int preLabel = 0; preLabel < allLabel.length; preLabel++)
                {

                    allFeature[transitionFeatureIndex] = preLabel; //7个模板特征+1个转移特征
                    double score = score(allFeature, curLabel); //从preLabel到curLabel的转移 + 从x(特征向量)发射到curLabel 的得分

                    double curScore = scoreMatrix[_i_1][preLabel] + score;

                    if (maxScore < curScore)
                    {
                        maxScore = curScore;
                        preMatrix[i][curLabel] = preLabel;
                        scoreMatrix[_i][curLabel] = maxScore;
                    }
                }
            }

        }
    }

    int maxIndex = 0;
    double maxScore = scoreMatrix[(sentenceLength - 1) & 1][0];

    for (int index = 1; index < allLabel.length; index++)
    {
        if (maxScore < scoreMatrix[(sentenceLength - 1) & 1][index])
        {
            maxIndex = index;
            maxScore = scoreMatrix[(sentenceLength - 1) & 1][index];
        }
    }

    for (int i = sentenceLength - 1; i >= 0; --i)
    {
        guessLabel[i] = allLabel[maxIndex];
        maxIndex = preMatrix[i][maxIndex];
    }

    return maxScore;
}
```

### 5.3.4 基于结构化的中文分词

- **特征提取**

![image-20221101200038706](C:%5CUsers%5C%E4%B8%AD%E5%8D%97%E5%AE%89%E7%90%AA%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20221101200038706.png)

上文用到的8维featureArray的前7维就是以上特征模板提取出的特征向量。

- **特征裁剪与模型压缩**

线性模型学习到的特征其实非常稀疏，大部分是低频特征，权重绝对值非常小，对预测结果影响非常小。在这里可以做如下压缩处理：

![image-20221101200350852](C:%5CUsers%5C%E4%B8%AD%E5%8D%97%E5%AE%89%E7%90%AA%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20221101200350852.png)

- **模型调整与在线学习**

1. **用户字典**

 “与川普通电话”分成“与，川，普通，电话”（没有川普这个词，可以通过**加入用户字典并设置高优先级**解决，但带来一个问题，**其他仍需要按照原来习惯分类的也被分错，**如“四川普通话”应该分成“四川，普通话”结果被分成“四，川普，通话”）

2. **在线学习**

读入用户提供的标注样本进行增量训练，模型就能学到新知识，学习次数可以逐个尝试。

```python
segment.enableCustomDictionary(False)
for i in range(3):                                  # 学三遍
    segment.learn("人 与 川普 通电话")                # 在线学习接口的输入必须是标注样本
print(segment.seg("银川普通人与川普通电话讲四川普通话"))
print(segment.seg("首相与川普通话讨论四川普通高考"))

# Output：
# [银川, 普通人, 与, 川普, 通电话, 讲, 四川, 普通话]
# [首相, 与, 川普, 通话, 讨论, 四川, 普通, 高考] # 说明模型有泛化能力
```

<!--个人认为，这里学到新知识，是featureMap扩大了，次数高的话w也会提高，使得这部分这样分的得分score更大，从而达到了效果。-->

- **中文分词特征工程**

反复在线学习仍然学不到新知识，可以重载特征提取部分，进行特征工程。常用的特征如下：

![image-20221101201100316](C:%5CUsers%5C%E4%B8%AD%E5%8D%97%E5%AE%89%E7%90%AA%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20221101201100316.png)

以叠字为例提取特征：

```java
public static class MyCWSInstance extends CWSInstance
{
    @Override
    protected int[] extractFeature(String sentence, FeatureMap featureMap, int position)
    {
        int[] defaultFeatures = super.extractFeature(sentence, featureMap, position);
        // 先调用父类提取特征
        char preChar = position >= 1 ? sentence.charAt(position - 1) : '_';
        String myFeature = preChar == sentence.charAt(position) ? "Y" : "N"; // 叠字特征
        int id = featureMap.idOf(myFeature);
        if (id != -1)
        {// 将叠字特征放到默认特征向量的尾部
            int[] newFeatures = new int[defaultFeatures.length + 1];
            System.arraycopy(defaultFeatures, 0, newFeatures, 0, defaultFeatures.length);
            newFeatures[defaultFeatures.length] = id;
            return newFeatures;
        }
        return defaultFeatures;
    }
    ...
}
```

```java
    public static void main(String[] args) throws IOException
    {
        CWSTrainer trainer = new CWSTrainer()
        {
            @Override
            protected Instance createInstance(Sentence sentence, FeatureMap featureMap)
            {
                return createMyCWSInstance(sentence, featureMap);
            }
        };
        LinearModel model = trainer.train(MSR.TRAIN_PATH, MSR.MODEL_PATH).getModel();
        PerceptronSegmenter segmenter = new PerceptronSegmenter(model)
        {
            @Override
            protected Instance createInstance(Sentence sentence, FeatureMap featureMap)
            {
                return createMyCWSInstance(sentence, featureMap);
            }
        };
        System.out.println(segmenter.segment("叠字特征帮助识别张文文李冰冰"));
    }

// Output:[叠字,特征,帮助,识别,张文文,李冰冰]
```

<!--个人认为，添加了新的特征，丰富了featureMap，再训练出合适的w，就在一定程度上学到了叠字划分的依据，更容易以高分找到正确的路径。-->

# 第六章 条件随机场与序列标注

Softmax函数详解：[一文详解Softmax函数 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/105722023)

（下层应用有要求时，需要把score转化为[0,1]内的概率且概率之和为1，用到Softmax函数处理）

## 6.1 生成式与判别式模型![image-20221102161844602](C:%5CUsers%5C%E4%B8%AD%E5%8D%97%E5%AE%89%E7%90%AA%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20221102161844602.png)

生成式模型和判别式模型详解：[机器学习中的判别式模型和生成式模型 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/74586507)

我们之前学习的隐马尔可夫模型属于生成式模型。生成式模型有一个死穴：

![image-20221102162108142](C:%5CUsers%5C%E4%B8%AD%E5%8D%97%E5%AE%89%E7%90%AA%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20221102162108142.png)

为了克服以上两个问题，判别式模型跳过了p(x)，直接对条件概率p(y | x)建模。哪怕x内部存在再复杂的依赖关系，也不影响判别式模型对于y的判断。所以就能够放心大胆利用各种丰富的、有关联的特征。（因此感知机模型的准确率高于隐马尔可夫模型）

## 6.2 无向图与最大团

[CRF条件随机场的原理、例子、公式推导和应用 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/148813079)

![image-20221102162816862](C:%5CUsers%5C%E4%B8%AD%E5%8D%97%E5%AE%89%E7%90%AA%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20221102162816862.png)

![image-20221102170141060](C:%5CUsers%5C%E4%B8%AD%E5%8D%97%E5%AE%89%E7%90%AA%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20221102170141060.png)

![image-20221102163113916](C:%5CUsers%5C%E4%B8%AD%E5%8D%97%E5%AE%89%E7%90%AA%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20221102163113916.png)

## 6.3 条件随机场CRF

是一种给定输入随机变量x，求解条件概率p(y | x)的概率无向图模型。

### 6.3.1 线性链条件随机场

用于序列标注时特例化为线性链。

![image-20221102163418033](C:%5CUsers%5C%E4%B8%AD%E5%8D%97%E5%AE%89%E7%90%AA%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20221102163418033.png)

![image-20221102163459750](C:%5CUsers%5C%E4%B8%AD%E5%8D%97%E5%AE%89%E7%90%AA%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20221102163459750.png)

**结构化感知机和条件随机场的联系：**

![image-20221102163532496](C:%5CUsers%5C%E4%B8%AD%E5%8D%97%E5%AE%89%E7%90%AA%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20221102163532496.png)

即：预测算法也是维特比算法，只需要关注调剂随机场的训练算法。

### 6.3.2 CRF的训练

![image-20221102163811946](C:%5CUsers%5C%E4%B8%AD%E5%8D%97%E5%AE%89%E7%90%AA%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20221102163811946.png)

L1和L2正则化详解：[L1正则化和L2正则化的区别 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/352437358)

接下来：

- 梯度上升（用期望来记，详见P207-208）
- 凸优化算法来优化梯度上升的方向（详见P209，210）

### 6.3.3 对比于结构化感知机

**相同点：**

- 特征函数相同
- 权重向量相同
- 打分函数相同
- 预测算法相同
- 同属结构化学习

**不同点：**

最大的不同点在于**训练算法**。这是两年这准确率差异的唯一原因。

![image-20221102164731559](C:%5CUsers%5C%E4%B8%AD%E5%8D%97%E5%AE%89%E7%90%AA%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20221102164731559.png)

![image-20221102164909600](C:%5CUsers%5C%E4%B8%AD%E5%8D%97%E5%AE%89%E7%90%AA%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20221102164909600.png)

### 6.3.4 CRF用于中文分词

![image-20221102170357878](C:%5CUsers%5C%E4%B8%AD%E5%8D%97%E5%AE%89%E7%90%AA%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20221102170357878.png)

![image-20221102170428991](C:%5CUsers%5C%E4%B8%AD%E5%8D%97%E5%AE%89%E7%90%AA%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20221102170428991.png)

也有用于词性标注、命名实体识别的相关内容：

[CRF条件随机场的原理、例子、公式推导和应用 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/148813079)

## 6.4 完整的模型使用

[(20条消息) 使用 CRF 做中文分词_INotWant的博客-CSDN博客_crf分词](https://blog.csdn.net/kiss_xiaojie/article/details/93380104)

```python
CRFSegmenter = JClass('com.hankcs.hanlp.model.crf.CRFSegmenter')

TXT_CORPUS_PATH = my_cws_corpus()
TSV_CORPUS_PATH = TXT_CORPUS_PATH + ".tsv"
TEMPLATE_PATH = test_data_path() + "/cws-template.txt"
CRF_MODEL_PATH = test_data_path() + "/crf-cws-model"
CRF_MODEL_TXT_PATH = test_data_path() + "/crf-cws-model.txt"


def train_or_load(corpus_txt_path=TXT_CORPUS_PATH, model_txt_path=CRF_MODEL_TXT_PATH):
    if os.path.isfile(model_txt_path):  # 已训练，直接加载
        segmenter = CRFSegmenter(model_txt_path)
        return segmenter
    else:
        segmenter = CRFSegmenter()  # 创建空白分词器
        segmenter.convertCorpus(corpus_txt_path, TSV_CORPUS_PATH)  # 执行转换
        segmenter.dumpTemplate(TEMPLATE_PATH)  # 导出特征模板
        # 交给CRF++训练
        print("语料已转换为 %s ，特征模板已导出为 %s" % (TSV_CORPUS_PATH, TEMPLATE_PATH))
        print("请安装CRF++后执行 crf_learn -f 3 -c 4.0 %s %s %s -t" % (TEMPLATE_PATH, TSV_CORPUS_PATH, CRF_MODEL_PATH))
        print("或者执行移植版 java -cp %s com.hankcs.hanlp.model.crf.crfpp.crf_learn -f 3 -c 4.0 %s %s %s -t" % (
            HANLP_JAR_PATH, TEMPLATE_PATH, TSV_CORPUS_PATH, CRF_MODEL_PATH))


if __name__ == '__main__':
    segment = train_or_load()
    if segment:
        print(segment.segment("商品和服务"))
```

命令行执行过程：

```
Microsoft Windows [版本 10.0.19043.1826]
(c) Microsoft Corporation。保留所有权利。

C:\Users\中南安琪>e:

E:\>cd E:\NLP\crfpp-0.58\CRF++-0.58

E:\NLP\crfpp-0.58\CRF++-0.58>crf_learn -f 3 -c 4.0 template 4_train.data 4_model -t > 4_train.txt
encoder.cpp(340) [feature_index.open(templfile, trainfile)] feature_index.cpp(135) [ifs] open failed: template

E:\NLP\crfpp-0.58\CRF++-0.58>crf_learn
CRF++: Yet Another CRF Tool Kit
Copyright (C) 2005-2013 Taku Kudo, All rights reserved.

Usage: crf_learn [options] files
 -f, --freq=INT              use features that occuer no less than INT(default 1)
 -m, --maxiter=INT           set INT for max iterations in LBFGS routine(default 10k)
 -c, --cost=FLOAT            set FLOAT for cost parameter(default 1.0)
 -e, --eta=FLOAT             set FLOAT for termination criterion(default 0.0001)
 -C, --convert               convert text model to binary model
 -t, --textmodel             build also text model file for debugging
 -a, --algorithm=(CRF|MIRA)  select training algorithm
 -p, --thread=INT            number of threads (default auto-detect)
 -H, --shrinking-size=INT    set INT for number of iterations variable needs to  be optimal before considered for shrinking. (default 20)
 -v, --version               show the version and exit
 -h, --help                  show this help and exit


E:\NLP\crfpp-0.58\CRF++-0.58>crf_learn -f 3 -c 4.0 E:\NLP\pyhanlp-master\pyhanlp\static\data\test/cws-template.txt E:\NLP\pyhanlp-master\pyhanlp\static\data\test\my_cws_corpus.txt.tsv E:\NLP\pyhanlp-master\pyhanlp\static\data\test/crf-cws-model -t
CRF++: Yet Another CRF Tool Kit
Copyright (C) 2005-2013 Taku Kudo, All rights reserved.

reading training data:
Done!0.00 s

Number of sentences: 3
Number of features:  52
Number of thread(s): 8
Freq:                3
eta:                 0.00010
C:                   4.00000
shrinking size:      20
iter=0 terr=0.61111 serr=1.00000 act=52 obj=24.95330 diff=1.00000
iter=1 terr=0.11111 serr=0.33333 act=52 obj=17.03450 diff=0.31734
iter=2 terr=0.27778 serr=0.33333 act=52 obj=9.29805 diff=0.45416
iter=3 terr=0.11111 serr=0.33333 act=52 obj=7.95629 diff=0.14431
iter=4 terr=0.11111 serr=0.33333 act=52 obj=7.24583 diff=0.08929
iter=5 terr=0.11111 serr=0.33333 act=52 obj=6.85361 diff=0.05413
iter=6 terr=0.11111 serr=0.33333 act=52 obj=6.73101 diff=0.01789
iter=7 terr=0.11111 serr=0.33333 act=52 obj=6.70316 diff=0.00414
iter=8 terr=0.11111 serr=0.33333 act=52 obj=6.68622 diff=0.00253
iter=9 terr=0.11111 serr=0.33333 act=52 obj=6.68247 diff=0.00056
iter=10 terr=0.11111 serr=0.33333 act=52 obj=6.68231 diff=0.00002
iter=11 terr=0.11111 serr=0.33333 act=52 obj=6.68211 diff=0.00003
iter=12 terr=0.11111 serr=0.33333 act=52 obj=6.68210 diff=0.00000

Done!0.07 s


E:\NLP\crfpp-0.58\CRF++-0.58>
```

再次运行python文件得到：

```python
D:\Anaconda\envs\python37\python.exe E:/NLP/pyhanlp-master/tests/book/ch06/crfpp_train_hanlp_load.py
        
[商品, 和, 服务]
```

## 6.5 标准化测评与模型总对比

```python
标准化测评结果：
P:96.86 R:96.64 F1:96.75 OOV-R:71.54 IV-R:97.33
```

![image-20221102173135645](C:%5CUsers%5C%E4%B8%AD%E5%8D%97%E5%AE%89%E7%90%AA%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20221102173135645.png)