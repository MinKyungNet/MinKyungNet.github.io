---
layout: post
title: "RDD를 사용하여 word vector를 생성하기"
tags: [Word Vector, flatMap, sortByKey, reduceByKey, groupByKey, mapValues, ]
categories: [Big data analsis]
---

# spark를 사용하기 위한 준비
```pyhton
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
사람은 연설이나 에세이 등을 읽고 나면 그 내용이 무엇이고 무엇을 말하려고 하는지 알 수 있다.             
주제가 무엇이고 심지어는 숨겨진 행간도 이해 할 수 있다.            
그러나 컴퓨터는 문서를 읽고 의미를 파악해 내는 것이 쉽지 않다.            
그 대신 문서에 어떤 단어가 쓰였고, 많이 쓰인 단어가 무엇인지 알아내어 어떤 애용인지 알아내게 된다.           
이 경우 쓰인 단어의 빈도를 word vector라고 하며, 문서로 부터 이를 만들어 내는 작업이 필요하다.        

Bag of words 모델         
문서 또는 문장을 단어의 집합 "bags of words"으로 표현한다.    
단어가 쓰인 순서나 문법은 무시한다.      
다음 문장은 단어와 그 횟수로 표현할 수 있다.

# 해결
분석하려는 문서를 RDD로 만들고, map-reduce 알고리즘으로 단어를 셀 수 있다.       
Spark mllib 라이버러리에서는 자연어처리 기법을 제공하고 있다.    
TF-IDF, Word2Vec등을 사용하면 단어의 빈도 및 관련어 등을 분석해 낼 수 있다.      
다음 장에서 배우겠지만 데이터프레임 ml을 사용할 수도 있다.    
```pyhton
import os
wikiRdd = spark.sparkContext.textFile(os.path.join("data", "ds_spark_wiki.txt"))
```

# 단어 집합
단어들을 공백으로 분리하고 단어의 집합으로 만든다.         
리스트롤 모두 하나로 합치는 faltMap()함수를 사용한다.         
그 결과 PipelineRDD가 만들어진다.
```pyhton
words = wikiRdd.flatMap(lambda x : x.split())
```
공백으로 분리된 RDD를 collect()함수로 취합한다.          
단어의 갯수는 count()함수로 알 수 있다.          
collect()결과는 list로 만들어진다.     
모든 항목을 출력하려면 list의 인덱스 [:]을 사용한다.

```pyhton
words.count()

>>

72
```

```pyhton
words.collect()

>>

['Wikipedia',
 'Apache',
 'Spark',
 'is',
 'an',
 'open',
 'source',
 'cluster',
 'computing',
 'framework.',
 '아파치',
 '스파크는',
 '오픈',
 '소스',
 '클러스터',
 '컴퓨팅',
 '프레임워크이다.',
 'Apache',
 'Spark',
 'Apache',
 'Spark',
 'Apache',
 'Spark',
 'Apache',
 'Spark',
 '아파치',
 '스파크',
 '아파치',
 '스파크',
 '아파치',
 '스파크',
 '아파치',
 '스파크',
 'Originally',
 'developed',
 'at',
 'the',
 'University',
 'of',
 'California,',
 "Berkeley's",
 'AMPLab,',
 'the',
 'Spark',
 'codebase',
 'was',
 'later',
 'donated',
 'to',
 'the',
 'Apache',
 'Software',
 'Foundation,',
 'which',
 'has',
 'maintained',
 'it',
 'since.',
 'Spark',
 'provides',
 'an',
 'interface',
 'for',
 'programming',
 'entire',
 'clusters',
 'with',
 'implicit',
 'data',
 'parallelism',
 'and',
 'fault-tolerance.']
```

# 단어 빈도
이제 map-reduce를 같이 사용하여 단어를 세어 tuple로 만들어보자.        
flatMap()은 앞서 설명한 바와 같이 파일 텍스트를 공백으로 분리한다.       
map()함수는 모든 단어에 대해 소문자로 만들고,          
불필요한 구문(new lines, commas, periods)을 제거한 후 tuple로 만든다.       
즉 (단어, 1)구조로 만들어 같은 단어는 나중에 서로 더할 수 있게 만들어 놓는다.        

```pyhton
words.map(lambda x: (x, 1)).reduceByKey(lambda x, y : x+y).collect()

>>

[('Wikipedia', 1),
 ('Apache', 6),
 ('Spark', 7),
 ('is', 1),
 ('an', 2),
 ('open', 1),
 ('source', 1),
 ('cluster', 1),
 ('computing', 1),
 ('framework.', 1),
 ('아파치', 5),
 ('스파크는', 1),
 ('오픈', 1),
 ('소스', 1),
 ('클러스터', 1),
 ('컴퓨팅', 1),
 ('프레임워크이다.', 1),
 ('스파크', 4),
 ('Originally', 1),
 ('developed', 1),
 ('at', 1),
 ('the', 3),
 ('University', 1),
 ('of', 1),
 ('California,', 1),
 ("Berkeley's", 1),
 ('AMPLab,', 1),
 ('codebase', 1),
 ('was', 1),
 ('later', 1),
 ('donated', 1),
 ('to', 1),
 ('Software', 1),
 ('Foundation,', 1),
 ('which', 1),
 ('has', 1),
 ('maintained', 1),
 ('it', 1),
 ('since.', 1),
 ('provides', 1),
 ('interface', 1),
 ('for', 1),
 ('programming', 1),
 ('entire', 1),
 ('clusters', 1),
 ('with', 1),
 ('implicit', 1),
 ('data', 1),
 ('parallelism', 1),
 ('and', 1),
 ('fault-tolerance.', 1)]
```

```pyhton
wc = spark.sparkContext.textFile(os.path.join("data","ds_spark_wiki.txt"))\
    .flatMap(lambda x: x.split(' '))\
    .map(lambda x: (x.lower().rstrip().lstrip().rstrip(',').rstrip('.'), 1))
```
아직 단어별로 갯수를 계산하지 않았기 때문에, 모두 1인 값을 가진다.         
sortByKey()는 오름차순을 기본으로 정렬한다.

```pyhton
wc.sortByKey().collect()
```

# 빈도 집계
이제 단어의 개수를 합계내어 보자.          
아래 방법 모두 동일한 결과를 산출한다.    

reduceByKey(add)               
groupByKey().mapValues(sum)    
groupByKey().map(lambda (x, iter): (x, len(iter)))

# reduceByKey()
python의 연산자 add()함수를 사용해서 할 수 있다.     
operator.add()는 reduce()함수의 숫자 인자 x, y를 받아서 x+y연산을 한다.    
```pyhton
from operator import add
wc.reduceByKey(add).sortByKey().collect()

>>

[('amplab', 1),
 ('an', 2),
 ('and', 1),
 ('apache', 6),
 ('at', 1),
 ("berkeley's", 1),
 ('california', 1),
 ('cluster', 1),
 ('clusters', 1),
 ('codebase', 1),
 ('computing', 1),
 ('data', 1),
 ('developed', 1),
 ('donated', 1),
 ('entire', 1),
 ('fault-tolerance', 1),
 ('for', 1),
 ('foundation', 1),
 ('framework', 1),
 ('has', 1),
 ('implicit', 1),
 ('interface', 1),
 ('is', 1),
 ('it', 1),
 ('later', 1),
 ('maintained', 1),
 ('of', 1),
 ('open', 1),
 ('originally', 1),
 ('parallelism', 1),
 ('programming', 1),
 ('provides', 1),
 ('since', 1),
 ('software', 1),
 ('source', 1),
 ('spark', 7),
 ('the', 3),
 ('to', 1),
 ('university', 1),
 ('was', 1),
 ('which', 1),
 ('wikipedia', 1),
 ('with', 1),
 ('소스', 1),
 ('스파크', 4),
 ('스파크는', 1),
 ('아파치', 5),
 ('오픈', 1),
 ('컴퓨팅', 1),
 ('클러스터', 1),
 ('프레임워크이다', 1)]
```

또는 add 대신에 reduceByKey()에 lambda함수를 사용해도 된다.

```pyhton
wc.reduceByKey(lambda x, y : x + y).sortByKey().collect()
```

# groupByKey(), mapValues()
groupByKey()는 단어키로 동일한 단어는 집단화한다.       
집단화하면 PairRdd가 되고 즉, key-value쌍으로 구성된다.         
mapValeus()는 각 쌍의 value에 대해서 sum연산을 한다.

```pyhton
wc.groupByKey().mapValues(sum).sortByKey().collect()
```

```pyhton
wc.groupByKey().map(lambda x : (x[0], len(x[1]))).sortByKey().collect()
```

# groupByKey(), map, len
mapValues()를 사용하지 않고 map()으로 len()갯수를 세어도 동일한 결과를 얻을 수 있다.       
단어별로 1개씩 만들어 놓았으므로 len()으로 세면 단어의 개수의 합계가된다.       
```pyhton
wc.groupByKey().map(lambda x : (x[0], len(x[1]))).sortByKey().collect()
```

# 줄로 구분하여 단어 빈도
flatMap()을 사용하지 않으면 줄로 구분하여 단어 빈도를 셀 수 있다.
```pyhton
wc_line = spark.sparkContext.textFile("data/ds_spark_wiki.txt")\
    .map(lambda x : x.replace(',', ' ').replace('.', ' ').replace('-', ' ').lower())\
    .map(lambda x : x.split())\
    .map(lambda x : [(i, 1) for i in x])
```

```pyhton
wc.sortByKey().collect()
```

# TF(Term Frequency)
단어빈도를 계산하기 위해 HashingTF를 사용할 수 있다.           
단어ID로 Hash 알고리즘에 따라 무작위 번호를 생성하고,          
단어 빈도를 생성한다.

# 단어 분리해서 RDD생성
```pyhton
wikiRdd3 = spark.sparkContext.textFile("data/ds_spark_wiki.txt")\
    .map(lambda x : x.split())
```
RDD는 mllib라이브러리를 사용한다.       
여기의 Hashing TF를 사용한다.       
transform()함수를 사용하여 RDD를 단어빈도 구조로 변환한다.       

```pyhton
from pyspark.mllib.feature import HashingTF

hashingTF = HashingTF()
tf = hashingTF.transform(wikiRdd3)
tf.collect()

>>

[SparseVector(1048576, {1026674: 1.0}),
 SparseVector(1048576, {148618: 1.0, 183975: 1.0, 216207: 1.0, 261052: 1.0, 617454: 1.0, 696349: 1.0, 721336: 1.0, 816618: 1.0, 897662: 1.0}),
 SparseVector(1048576, {60386: 1.0, 177421: 1.0, 568609: 1.0, 569458: 1.0, 847171: 1.0, 850510: 1.0, 1040679: 1.0}),
 SparseVector(1048576, {261052: 4.0, 816618: 4.0}),
 SparseVector(1048576, {60386: 4.0, 594754: 4.0}),
 SparseVector(1048576, {21980: 1.0, 70882: 1.0, 274690: 1.0, 357784: 1.0, 549790: 1.0, 597434: 1.0, 804583: 1.0, 829803: 1.0, 935701: 1.0}),
 SparseVector(1048576, {154253: 1.0, 261052: 1.0, 438276: 1.0, 460085: 1.0, 585459: 1.0, 664288: 1.0, 816618: 1.0, 935701: 2.0, 948143: 1.0, 1017889: 1.0}),
 SparseVector(1048576, {270017: 1.0, 472985: 1.0, 511771: 1.0, 718483: 1.0, 820917: 1.0}),
 SparseVector(1048576, {34116: 1.0, 87407: 1.0, 276491: 1.0, 348943: 1.0, 482882: 1.0, 549350: 1.0, 721336: 1.0, 816618: 1.0, 1025622: 1.0}),
 SparseVector(1048576, {1769: 1.0, 151357: 1.0, 500659: 1.0, 547760: 1.0, 979482: 1.0})]
```

