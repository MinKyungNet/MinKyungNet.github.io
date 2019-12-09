---
layout: post
title: "DataFrame Transform"
tags: [Bag of Words, Tokenizer, Stopwords, CountVectorizer, TF-IDF, Word2Vec, Pipeline]
categories:[Big data analsis]
---

```python
import os
import sys
os.environ["SPARK_HOME"]=os.path.join(os.path.expanduser("~"),"spark-2.4.4-bin-hadoop2.7")
os.environ["PYLIB"]=os.path.join(os.environ["SPARK_HOME"],'python','lib')
sys.path.insert(0,os.path.join(os.environ["PYLIB"],'py4j-0.10.7-src.zip'))
sys.path.insert(0,os.path.join(os.environ["PYLIB"],'pyspark.zip'))
import pyspark
myConf=pyspark.SparkConf()
spark = pyspark.sql.SparkSession.builder\
    .master("local")\
    .appName("myApp")\
    .config(conf=myConf)\
    .getOrCreate()
```


```python
fsvm = os.path.join(os.environ["SPARK_HOME"], 'data', 'mllib', 'sample_libsvm_data.txt')
dfsvm = spark.read.format("libsvm").load(fsvm)
```

# 통계
```python
from pyspark.mllib.stat import Statistics
parallelData = spark.sparkContext.parallelize([1.0, 2.0, 5.0, 4.0, 3.0, 3.3, 5.5])
testResult = Statistics.kolmogorovSmirnovTest(parallelData, "norm")
print(testResult)

>>>

Kolmogorov-Smirnov test summary:
degrees of freedom = 0 
statistic = 0.841344746068543 
pValue = 5.06089025353873E-6 
Very strong presumption against null hypothesis: Sample follows theoretical distribution.
```

# S-3 : 평균, 표준편차와 같은 기본 통계 값을 구한다

### 문제
균등 분포 및 정규분포를 무작위로 생성해 기본통계 값을 계산해 보자.

### 해결
무작위는 발생빈도가 어느 쪽에 치우치지 않는다. Spark에서 무작위로 균등 분포 및 정규분포를 생성하고, 기본통계를 계산한다.

### 분포 생성
DataFrame에서 제공하는 통계기능을 사용해본다.
```python
df = spark.range(0, 10)
df.show()
df.select('id')

>>>

+---+
| id|
+---+
|  0|
|  1|
|  2|
|  3|
|  4|
|  5|
|  6|
|  7|
|  8|
|  9|
+---+

DataFrame[id: bigint]
```


```python
from pyspark.sql.functions import rand, randn
colUniform = rand(seed=10).alias("uniform")
colNormal = randn(seed=27).alias("normal")
df3=df.select("id", colUniform, colNormal)
df3.show()

>>>

+---+-------------------+-------------------+
| id|            uniform|             normal|
+---+-------------------+-------------------+
|  0|0.41371264720975787| 0.5888539012978773|
|  1| 0.7311719281896606| 0.8645537008427937|
|  2| 0.9031701155118229| 1.2524569684217643|
|  3|0.09430205113458567| -2.573636861034734|
|  4|0.38340505276222947| 0.5469737451926588|
|  5| 0.5569246135523511|0.17431283601478723|
|  6| 0.4977441406613893|-0.7040284633147095|
|  7| 0.2076666106201438| 0.4637547571868822|
|  8| 0.9571919406508957|  0.920722532496133|
|  9| 0.7429395461204413|-1.4353459012380192|
+---+-------------------+-------------------+
```

### 기본 통계

주사위는 이산균등분포의 가장 대표적인 예이다. 각 숫자가 나올 확률을 1/6이다. 정규분포는 평균 0을 중심으로 빈도가 몰려있어 표준편차 만큼 퍼진 특징을 가진다. 각 컬럼별로 통계값을 계산할 수 있다.
```python
df3.describe().show()

>>>

+-------+------------------+-------------------+--------------------+
|summary|                id|            uniform|              normal|
+-------+------------------+-------------------+--------------------+
|  count|                10|                 10|                  10|
|   mean|               4.5| 0.5488228646413278|0.009861721586543392|
| stddev|3.0276503540974917| 0.2856822245344392|  1.2126061129356596|
|    min|                 0|0.09430205113458567|  -2.573636861034734|
|    max|                 9| 0.9571919406508957|  1.2524569684217643|
+-------+------------------+-------------------+--------------------+
```

### freqItems()
abc 세 컬럼을 생성한다. 홀수 행이면 1,2,3으로 짝수 행이면 다른 수열로 DataFrame을 생성해보자. 이 데이터에 대해 60%이상 발생한 행을 출력해보자.
```python
df = spark.createDataFrame([(1,2,3) if i % 2 == 0 else (i, 2*i, i%4) for i in range(100)])
df.show()

>>>

+---+---+---+
| _1| _2| _3|
+---+---+---+
|  1|  2|  3|
|  1|  2|  1|
|  1|  2|  3|
|  3|  6|  3|
|  1|  2|  3|
|  5| 10|  1|
|  1|  2|  3|
|  7| 14|  3|
|  1|  2|  3|
|  9| 18|  1|
|  1|  2|  3|
| 11| 22|  3|
|  1|  2|  3|
| 13| 26|  1|
|  1|  2|  3|
| 15| 30|  3|
|  1|  2|  3|
| 17| 34|  1|
|  1|  2|  3|
| 19| 38|  3|
+---+---+---+
only showing top 20 rows
```

# S.5 DataFrame 변환
dataFrame으로 만들어진 데이터를 변환해보자. 이러한 작업이 필요한 이유는 기계학습에 넘겨줄 입력데이터를 형식에 맞추어야하기 때문이다. 데이터는 형식에 맞게 변환되고, 군집화, 회귀 분석, 분류, 추천 모델 등에 입력으로 사용된다. 물론 데이터는 '일련의 수' 또는 '텍스트'로 구성된다. 이런 데이터로부터 특징을 추출하여 feature vectors를 구성한다.지도학습을 하는 경우에는 class또는 label값이 필요하다.           

## S.5.1 텍스트 변환
### Bag of Words 모델
텍스트를 단어의 집합, 'bag of words'으로 구성된다고 보며, 단어의 순서는 의미를 가지지 않는다.        
단계     
단계 1: 단어로 분할 Tokenization       
단계 2: 정리, 불필요, 오류 정리              
단계 3: 불용어 stopwords 제거         
단계 4: 어간 추출 stemming
단계 5: 계량화, word vector로 만든다. 있다 없아, 단어빈도, TF-IDF, dense, sparse

### S.5.2 Python을 사용한 단어 빈도 계산
```python
# Let it be lyrics
doc=[
    "When I find myself in times of trouble",
    "Mother Mary comes to me",
    "Speaking words of wisdom, let it be",
    "And in my hour of darkness",
    "She is standing right in front of me",
    "Speaking words of wisdom, let it be",
    "Let it be",
    "Let it be",
    "Let it be",
    "Let it be",
    "Whisper words of wisdom, let it be"
]
```

문서, 문장, 단어의 계층을 먼저 이해해야 한다. 문서는 문장으로 구성되어 있고, 문장은 단어로 구성되어 있다. 따라서 첫째 반복문은 문서의 각 문장에 대해, 단어로 분리하고 있다. 그 다음 반복문은 각 단어에 대해 빈도를 계산한다. 각 단어가 키가 되는데, 키가 존재하면 빈도를 증가하고, 존재하지 않으면 새로운 키를 생성한다.

```python
d = {}
for sentence in doc:
    words = sentence.split()
    for word in words:
        if word in d:
            d[word] += 1
        else:
            d[word] = 1
```

앞서 단어 빈도는 dictinary d에 저장하였다. dictionary는 키, 빈도의 쌍으로 저장되어 있어서 iteritems()으로 읽어낼 수 있다.

```python
for k, v in d.items():
    print(k, v)

>>>

When 1
I 1
find 1
myself 1
in 3
times 1
of 6
trouble 1
Mother 1
Mary 1
comes 1
to 1
me 2
Speaking 2
words 3
wisdom, 3
let 3
it 7
be 7
And 1
my 1
hour 1
darkness 1
She 1
is 1
standing 1
right 1
front 1
Let 4
Whisper 1
```



```python
print(d)

>>>

{'When': 1, 'I': 1, 'find': 1, 'myself': 1, 'in': 3, 'times': 1, 'of': 6, 'trouble': 1, 'Mother': 1, 'Mary': 1, 'comes': 1, 'to': 1, 'me': 2, 'Speaking': 2, 'words': 3, 'wisdom,': 3, 'let': 3, 'it': 7, 'be': 7, 'And': 1, 'my': 1, 'hour': 1, 'darkness': 1, 'She': 1, 'is': 1, 'standing': 1, 'right': 1, 'front': 1, 'Let': 4, 'Whisper': 1}
```

### 5.3 Spark
텍스트를 2차원 배열로 만들어, DataFrame을 생성한다.
```python
doc2d=[
    ["When I find myself in times of trouble"],
    ["Mother Mary comes to me"],
    ["Speaking words of wisdom, let it be"],
    ["And in my hour of darkness"],
    ["She is standing right in front of me"],
    ["Speaking words of wisdom, let it be"],
    [u"우리 Let it be"],
    [u"나 Let it be"],
    [u"너 Let it be"],
    ["Let it be"],
    ["Whisper words of wisdom, let it be"]
]

myDf = spark.createDataFrame(doc2d, ['sent'])

myDf.show()

>>>

+--------------------+
|                sent|
+--------------------+
|When I find mysel...|
|Mother Mary comes...|
|Speaking words of...|
|And in my hour of...|
|She is standing r...|
|Speaking words of...|
|      우리 Let it be|
|        나 Let it be|
|        너 Let it be|
|           Let it be|
|Whisper words of ...|
+--------------------+
```

# S.5.4 Tokenizer
Tokenizer는 문장을 단어로 분리한다. 분리하는 기준은 whitespace로 공백, tab, cr, new line등이 해당된다.입력은 sent로 출력은 words로 한다.
```python
from pyspark.ml.feature import Tokenizer
tokenizer = Tokenizer(inputCol = "sent", outputCol = "words")

tokDf = tokenizer.transform(myDf)

tokDf.show(3)

>>>

+--------------------+--------------------+
|                sent|               words|
+--------------------+--------------------+
|When I find mysel...|[when, i, find, m...|
|Mother Mary comes...|[mother, mary, co...|
|Speaking words of...|[speaking, words,...|
+--------------------+--------------------+
only showing top 3 rows

```


# S.5.5 RegTokenizer
tokenizer는 white space로 분리하지만, RegexTokenizer는 단어를 분리하기 위해 정규 표현식을 적용할 수 있다. 정규표현식을 사용하여 분리하거나 특정 패턴을 추출할 수 있다. 공백으로 분리할 경우 간단히 정규표현식 \s 패턴을 적용할 수 있다. 한글에는 \w패턴이 적용되지 않는다.
```python
from pyspark.ml.feature import RegexTokenizer
re = RegexTokenizer(inputCol = "sent", outputCol = "wordReg", pattern = "\\s+")

reDf = re.transform(myDf)
reDf.show()
>>>

+--------------------+--------------------+
|                sent|             wordReg|
+--------------------+--------------------+
|When I find mysel...|[when, i, find, m...|
|Mother Mary comes...|[mother, mary, co...|
|Speaking words of...|[speaking, words,...|
|And in my hour of...|[and, in, my, hou...|
|She is standing r...|[she, is, standin...|
|Speaking words of...|[speaking, words,...|
|      우리 Let it be| [우리, let, it, be]|
|        나 Let it be|   [나, let, it, be]|
|        너 Let it be|   [너, let, it, be]|
|           Let it be|       [let, it, be]|
|Whisper words of ...|[whisper, words, ...|
+--------------------+--------------------+
```

# S.5.6 Stopwords
텍스트를 분리하고 나면, 별 의미가 없거나 쓸모가 없는 단어들이 존재한다. 불필요한 단어들을 불용어라고하며 입력데이터에서 제거하도록한다.
```python
from pyspark.ml.feature import StopWordsRemover
stop = StopWordsRemover(inputCol = "wordReg", outputCol = "nostops")

stopwords = list()
_stopwords = stop.getStopWords()
for e in _stopwords:
    stopwords.append(e)
    
_mystopwords = [u"나", u"너", u"우리"]
for e in _mystopwords:
    stopwords.append(e)
stop.setStopWords(stopwords)

for e in stop.getStopWords():
    print(e)
    
>>>

i
me
my
myself
we
our
ours
ourselves
you
your
y
...
```

# S.5.7 CountVectorizer
CountVectorzier는 텍스트를 입력해서, word vector를 출력한다. 우선 Tokenizer를 사용해서 단어로 분리하고 난 후 사용한다.         
* minDF
    - 소수점은 비율, 사용된 문서 수 / 전체 문서 수
        - 정수는 사용된 문서 수, 단어가 몇 개의 문서에 사용되어야 하는지
* 입력 : a collection of text documents
* 출력 : word vector (sparse) vocabulary x TF
```python
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(stop_words='english')
print(vectorizer.fit_transform(doc))

>>>

  (0, 10)	1
  (0, 9)	1
  (1, 0)	1
  (1, 4)	1
  (1, 5)	1
  (2, 3)	1
  (2, 12)	1
  (2, 13)	1
  (2, 7)	1
  (3, 1)	1
  (3, 2)	1
  (4, 6)	1
  (4, 8)	1
  (5, 3)	1
  (5, 12)	1
  (5, 13)	1
  (5, 7)	1
  (6, 3)	1
  (7, 3)	1
  (8, 3)	1
  (9, 3)	1
  (10, 11)	1
  (10, 3)	1
  (10, 12)	1
  (10, 13)	1
```


```python
print(vectorizer.vocabulary_)

>>>

{'times': 9, 'trouble': 10, 'mother': 5, 'mary': 4, 'comes': 0, 'speaking': 7, 'words': 13, 'wisdom': 12, 'let': 3, 'hour': 2, 'darkness': 1, 'standing': 8, 'right': 6, 'whisper': 11}
```

Spark CountVectorizer
```python
from pyspark.ml.feature import CountVectorizer
cv = CountVectorizer(inputCol="nostops", outputCol="cv", vocabSize=30, minDF=1.0)

cvModel = cv.fit(stopDf)

cvModel = cv.fit(stopDf)

cvDf = cvModel.transform(stopDf)

cvDf.show(3)

>>>

+--------------------+--------------------+--------------------+
|                sent|             nostops|                  cv|
+--------------------+--------------------+--------------------+
|When I find mysel...|[find, times, tro...|(16,[5,6,8],[1.0,...|
|Mother Mary comes...|[mother, mary, co...|(16,[10,13,14],[1...|
|Speaking words of...|[speaking, words,...|(16,[0,1,2,3],[1....|
|And in my hour of...|    [hour, darkness]|(16,[7,9],[1.0,1.0])|
|She is standing r...|[standing, right,...|(16,[4,12,15],[1....|
|Speaking words of...|[speaking, words,...|(16,[0,1,2,3],[1....|
|      우리 Let it be|               [let]|      (16,[0],[1.0])|
|        나 Let it be|               [let]|      (16,[0],[1.0])|
|        너 Let it be|               [let]|      (16,[0],[1.0])|
|           Let it be|               [let]|      (16,[0],[1.0])|
|Whisper words of ...|[whisper, words, ...|(16,[0,1,2,11],[1...|
+--------------------+--------------------+--------------------+
```

# S.5.8 TF-IDF
Term Frequency - Inverse Document Frequency를 계산한다.    
이를 위해서는 우선 Tokenizer를 사용하여 문장을 단어로 분리해 놓아야한다.    
HashingTF를 사용하여 'word vector'를 계산한다. HashingTF은 hash함수에 따라 단어의 고유 번호를 생성하며, hash고유의 충돌 가능성을 줄이기 위해, 단어수를 제안할 수 있다. 그리고 IDF를 계산하고 TF-IDF를 계산한다.

### S.5.8.1 TF-IDF 계산
```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_df = 1.0, min_df=1, stop_words = 'english', norm = None)

print(vectorizer.fit_transform(doc))
>>>

  (0, 9)	2.791759469228055
  (0, 10)	2.791759469228055
  (1, 5)	2.791759469228055
  (1, 4)	2.791759469228055
  (1, 0)	2.791759469228055
  (2, 7)	2.386294361119891
  (2, 13)	2.09861228866811
  (2, 12)	2.09861228866811
  (2, 3)	1.4054651081081644
  (3, 2)	2.791759469228055
  (3, 1)	2.791759469228055
  (4, 8)	2.791759469228055
  (4, 6)	2.791759469228055
  (5, 7)	2.386294361119891
  (5, 13)	2.09861228866811
  (5, 12)	2.09861228866811
  (5, 3)	1.4054651081081644
  (6, 3)	1.4054651081081644
  (7, 3)	1.4054651081081644
  (8, 3)	1.4054651081081644
  (9, 3)	1.4054651081081644
  (10, 13)	2.09861228866811
  (10, 12)	2.09861228866811
  (10, 3)	1.4054651081081644
  (10, 11)	2.791759469228055
```

### S.5.8.3 spark를 사용한 TF-IDF
```python
from pyspark.ml.feature import HashingTF, IDF
hashTF = HashingTF(inputCol = "nostops", outputCol="hash", numFeatures=50)

hashDf = hashTF.transform(stopDf)

hashDf.show()

>>>

+--------------------+--------------------+--------------------+--------------------+
|                sent|             wordReg|             nostops|                hash|
+--------------------+--------------------+--------------------+--------------------+
|When I find mysel...|[when, i, find, m...|[find, times, tro...|(50,[10,24,43],[1...|
|Mother Mary comes...|[mother, mary, co...|[mother, mary, co...|(50,[1,21,24],[1....|
|Speaking words of...|[speaking, words,...|[speaking, words,...|(50,[9,12,14,41],...|
|And in my hour of...|[and, in, my, hou...|    [hour, darkness]|(50,[23,27],[1.0,...|
|She is standing r...|[she, is, standin...|[standing, right,...|(50,[24,43,46],[1...|
|Speaking words of...|[speaking, words,...|[speaking, words,...|(50,[9,12,14,41],...|
|      우리 Let it be| [우리, let, it, be]|               [let]|     (50,[14],[1.0])|
|        나 Let it be|   [나, let, it, be]|               [let]|     (50,[14],[1.0])|
|        너 Let it be|   [너, let, it, be]|               [let]|     (50,[14],[1.0])|
|           Let it be|       [let, it, be]|               [let]|     (50,[14],[1.0])|
|Whisper words of ...|[whisper, words, ...|[whisper, words, ...|(50,[9,14,15,41],...|
+--------------------+--------------------+--------------------+--------------------+
```


```python
idf = IDF(inputCol="hash", outputCol="idf")

idfModel = idf.fit(hashDf)
idfDf = idfModel.transform(hashDf)

for e in idfDf.select("nostops", 'hash').take(10):
    print(e)
    
>>>

Row(nostops=['find', 'times', 'trouble'], hash=SparseVector(50, {10: 1.0, 24: 1.0, 43: 1.0}))
Row(nostops=['mother', 'mary', 'comes'], hash=SparseVector(50, {1: 1.0, 21: 1.0, 24: 1.0}))
Row(nostops=['speaking', 'words', 'wisdom,', 'let'], hash=SparseVector(50, {9: 1.0, 12: 1.0, 14: 1.0, 41: 1.0}))
Row(nostops=['hour', 'darkness'], hash=SparseVector(50, {23: 1.0, 27: 1.0}))
Row(nostops=['standing', 'right', 'front'], hash=SparseVector(50, {24: 1.0, 43: 1.0, 46: 1.0}))
Row(nostops=['speaking', 'words', 'wisdom,', 'let'], hash=SparseVector(50, {9: 1.0, 12: 1.0, 14: 1.0, 41: 1.0}))
Row(nostops=['let'], hash=SparseVector(50, {14: 1.0}))
Row(nostops=['let'], hash=SparseVector(50, {14: 1.0}))
Row(nostops=['let'], hash=SparseVector(50, {14: 1.0}))
Row(nostops=['let'], hash=SparseVector(50, {14: 1.0}))
```

# S.5.10 Word2Vec
```python
from pyspark.ml.feature import Word2Vec
word2Vec = Word2Vec(vectorSize=3, minCount=0, inputCol="words", outputCol="w2v")
model = word2Vec.fit(tokDf)
w2vDf = model.transform(tokDf)
for e in w2vDf.select("w2v").take(3):
    print(e)

>>>

Row(w2v=DenseVector([-0.0194, 0.0106, -0.0368]))
Row(w2v=DenseVector([0.0381, 0.0235, -0.0119]))
Row(w2v=DenseVector([-0.0511, 0.0448, -0.0002]))
```

# 5.14 Pipeline
pipeline은 여러 Esimator를 묶은 Esimator를 반환한다. 단계적으로 Estimator를 적용하기 위해 사용한다.
```python

from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.classification import LogisticRegression

df = spark.createDataFrame([
    (0, "a b c d e spark", 1.0),
    (1, "b d", 0.0),
    (2, "spark f g h", 1.0),
    (3, "hadoop mapreduce", 0.0),
    (4, "my dog has flea problems. help please.",0.0)
    ], ["id", "text", "label"])

tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
lr = LogisticRegression(maxIter=10, regParam=0.01)
pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])

model = pipeline.fit(df)
myDf = model.transform(df)
```

# 문제 S-4: 연설문을 기계학습하기 위해 변환
한 글자 단어를 제외하고, 단어의 TF=IDF를 계산해서 feature로 구성한다.
```python
import os

from pyspark.sql.types import StructType, StructField, StringType
police = spark.read\
    .option('header', 'true')\
    .option('delimiter', ' ')\
    .option('inferSchema', 'true')\
    .schema(
        StructType([
            StructField("sent", StringType(),)
        ])
    )\
    .text(os.path.join("data", "20191021_policeAddress.txt"))

police.show(5, False)

>>>

+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|sent                                                                                                                                                                                                                                        |
+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|존경하는 국민 여러분, 경찰관 여러분, 일흔네 돌 ‘경찰의 날’입니다.                                                                                                                                                                           |
|                                                                                                                                                                                                                                            |
|국민의 안전을 위해 밤낮없이 애쓰시는 전국의 15만 경찰관 여러분께 먼저 감사를 드립니다. 전몰·순직 경찰관들의 고귀한 희생에 경의를 표합니다. 유가족 여러분께 위로의 마음을 전합니다.                                                          |
|                                                                                                                                                                                                                                            |
|오늘 홍조근정훈장을 받으신 중앙경찰학교장 이은정 치안감님, 근정포장을 받으신 광주남부경찰서 김동현 경감님을 비롯한 수상자 여러분께 각별한 축하와 감사를 드립니다. 또한 경찰 영웅으로 추서되신 차일혁, 최중락님께 국민의 사랑을 전해드립니다.|
+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
only showing top 5 rows
```


```python
from pyspark.ml.feature import Tokenizer
tokenizer = Tokenizer(inputCol = "sent", outputCol="words")
tokDf = tokenizer.transform(police)

from pyspark.ml.feature import StopWordsRemover
stop = StopWordsRemover(inputCol = "words", outputCol = "nostops")
stop.setStopWords([u"들", u"너", u"우리"])
_mystopwords= [u"들", u"너", u"우리"]
for e in _mystopwords:
    stopwords.append(e)
stop.setStopWords(stopwords)

stopDf=stop.transform(tokDf)
stopDf.show()
>>>

+----------------------------------+--------------------------------+--------------------------------+
|                              sent|                           words|                         nostops|
+----------------------------------+--------------------------------+--------------------------------+
|  존경하는 국민 여러분, 경찰관 ...|   [존경하는, 국민, 여러분,, ...|   [존경하는, 국민, 여러분,, ...|
|                                  |                              []|                              []|
| 국민의 안전을 위해 밤낮없이 애...|  [국민의, 안전을, 위해, 밤낮...|  [국민의, 안전을, 위해, 밤낮...|
|                                  |                              []|                              []|
|오늘 홍조근정훈장을 받으신 중앙...|[오늘, 홍조근정훈장을, 받으신...|[오늘, 홍조근정훈장을, 받으신...|
|                                  |                              []|                              []|
|           사랑하는 경찰관 여러분,|     [사랑하는, 경찰관, 여러분,]|     [사랑하는, 경찰관, 여러분,]|
|                                  |                              []|                              []|
|여러분의 헌신적 노력으로 우리의...| [여러분의, 헌신적, 노력으로,...| [여러분의, 헌신적, 노력으로,...|
|                                  |                              []|                              []|
| 치안의 개선은 국민의 체감으로 ...|  [치안의, 개선은, 국민의, 체...|  [치안의, 개선은, 국민의, 체...|
|                                  |                              []|                              []|
| 한국을 찾는 외국 관광객들도 우...|  [한국을, 찾는, 외국, 관광객...|  [한국을, 찾는, 외국, 관광객...|
|                                  |                              []|                              []|
|   올해는 ‘경찰의 날’에 맞춰 국...|    [올해는, ‘경찰의, 날’에, ...|    [올해는, ‘경찰의, 날’에, ...|
|                                  |                              []|                              []|
|         자랑스러운 경찰관 여러분,|   [자랑스러운, 경찰관, 여러분,]|   [자랑스러운, 경찰관, 여러분,]|
|                                  |                              []|                              []|
| 경찰헌장은 “나라와 겨레를 위하...| [경찰헌장은, “나라와, 겨레를...| [경찰헌장은, “나라와, 겨레를...|
|                                  |                              []|                              []|
+----------------------------------+--------------------------------+--------------------------------+
only showing top 20 rows
```
