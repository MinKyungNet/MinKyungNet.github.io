---
layout: post
title: "vectors, labeled point"
tags: [vector, labeled point]
categories: [Big data analsis]
---

# 3. Vectors
spark에서는 Vector, Labeled Point, Matrix를 사용할 수 있다.         

Vector : numpy vector과 같은 기능을 한다. dense와 sparse vector로 구분한다.             
Labeled Point : 분류를 의미하는 클래스 또는 label과 속성 features이 묶인 구조로서, supervised learning에 사용된다.         
Matrix : numpy matrix와 같은 특징을 가진다.           

이러한 데이터 타잎은 Spark의 ml, mllib패키지 별로 제공되므로, 식별하여 사용한다.            
ml패키지를 사용할 경우에는 자신의 pyspark.ml.linalg.Vector 등을 사용해야한다.      
mllib도 마찬가지이다.        

mllib : RDD API를 제공한다.           
ml : DataFrame API를 제공한다.    

# 3.1 vectors
Vector는 dense와 sparse로 구분할 수 있다.       
sparse는 실제 값이 없는 요소, '0'을 제거하여 만든 vector이다.     
Spark가 효율적으로 메모리를 사용하기 위해 자동으로 변환하여 사용하기도 한다.       
Spark에서 type field(1 바이트 길이)를 통해 식별한다.(0 : sparse, 1 : dense)            

예를 들어, 다음은 dense vector이다.     
(160, 69, 24)        
이를 sparse vector로 표현하면, 각 컬럼별 해당하는 값을 적는다.       
값이 없는 요소가 없으니 더 복잡해 보인다.     
(3, [0, 1, 2], [160.0, 69.0, 24.0])
dense vector : 모든 행열 값을 가지고 있다. 빈 값이 별로 없는 경우, (160, 69, 24), numpy array, Python list를 입력으로 사용     
sparse vector : 인덱스 및 값의 배열을 별도로 가진다, 빈 값이 많은 경우 사용, (3, [0, 1, 2], [160.0, 69.0, 24.0])컬럼 3개, 값이 있는 컬럼, 값, Vectors.sparse(), SciPy's csc_matrix          

### Dense vectors     
numpy array를 사용해도 dense vector를 만들 수 있다.       
Spark 내부적으로 numpy.array를 사용하고 있다.         

```python
import numpy as np
dv = np.array([1.0, 2.1, 3])
```
Spark에서는 RDD mllib, DataFrame ml의 Vectors를 사용하여 dense vector를 만들 수 있다.    
```python
from pyspark.mllib.linalg import Vectors

dv1mllib = Vectors.dense([1.0, 2.1, 3])
print(dv1mllib, type(dv1mllib)
```

```python
from pyspark.ml.linalg import Vectors
dv1ml = Vectors.dense([1.0, 2.1, 3])
print(dv1ml)
```

dense vectors는 numpy array와 같은 특징을 가진다.     
인덱스로 값을 읽을 수 있다. 또한 반복문에서 사용할 수 있다.           

보통 벡터와 같이 product, dot, norm과 같은 벡터 연산을 할 수도 있다.      
결과 값은 numpy와 동일하다.        

### Sparse vectors      

sparse vectors는 값 중에 0이 포함된 경우 이를 생략한다.     
toArray()함수를 사용하면 sparse에서 dense로 벡터를 변환할 수 있다.     

```python
sv1 = Vectors.sparse(3, [1,2], [1.0, 3.0])
print(sv1.toArray())
```

# 3.2 Labeled Point

label, features로 구성   
분류 및 회귀분석에 사용되는 데이터 타잎이다.         
label : supervised learning에서 '구분 값'으로 사용한다. 데이터 타잎은 'Double'    
features : sparse, dense 모두 사용할 수 있다.         

label 1.0, features [1.0, 2.0, 3.0]으로 Labeled Point를 만들어 보자.     

```python
from pyspark.mllib.regressing import LabeledPoint
print(LabeledPoint(1.0, [1.0, 2.0, 3.0])
```

```python
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors

print(LabeledPoint(1992, Vectors.sparse(10, {0: 3.0, 1:5.5, 2: 10.0})))
```
서로 다른 패키지의 데이터타잎 mllib LabeledPoint와 ml Vectors를 혼용하면, 형변환 오류가 발생한다. 이러한 오류는 패치지를 혼용하지 않으면 된다.        

### DataFrame에서 Labeled Point

* python list에서 DataFrame생성
```python
p = [[1,[1.0,2.0,3.0]],[1,[1.1,2.1,3.1]],[0,[1.2,2.2,3.3]]]
trainDf=spark.createDataFrame(p)
trainDf.collect()
```

# Python list를 LabeledPoint로 생성하면, 'label'과 'features'의 명칭을 가지도록 생성된다.        

```python
from pyspark.mllib.regression import LabeledPoint
p = [LabeledPoint(1, [1.0, 2.0, 3.0]),
     LabeledPoint(1, [1.1, 2.1, 3.1]),
     LabeledPoint(0, [1.2, 2.2, 3.3])]
trainDf = spark.createDataFrame(p)
trainDf.collect()
```

mllib.linalg.Vector를 사용하여 DataFrame을 생성해보자.     
```python
from pyspark.mllib.linalg import Vectors

trainDf = spark.createDataFrame([
    (1.0, Vectors.dense([0.0, 1.1, 0.1])),
    (0.0, Vectors.dense([2.0, 1.0, 1.0])),
    (0.0, Vectors.dense([2.0, 1.3, 1.0])),
    (1.0, Vectors.dense([0.0, 1.2, 0.5]))],
    ["label", "features"])

trainDf.collect()
```


schema를 사용해서 DataFrame을 생성해보자         
* 'label'은 Double Type
* 'features'는 Vector Type

```python
from pyspark.mllib.linalg import SparseVector, VectorUDT
from pyspark.sql.types import StructType, StructField, DoubleType
_rdd = spark.sparkContext.parallelize([
    (0.0, SparseVector(4, {1: 1.0, 3: 5.5})),
    (1.0, SparseVecotr(4, {0: -1.0, 2: 0.5}))])
    
schema = StructType([
    StructField("label", DoubleType(), True),
    StructField("features", VectorUDT(), True)
])

trainDf = _rdd.toDF(schema)
trainDf.printSchema()
```

### sparse에서 dense vector로 변환
방금 생성한 trainDf는 sparse vector이다.          
사용자 함수 udf를 사용하여 sparse vector를 dense vector로 변환해보자.        
바로 변환할 수 있는 함수 toDense() 함수를 지원하지 않으므로,              
sparse vector를 toArray() 함수를 사용해서 dense vector로 변환한다.           

또 trainDf는 mllib RDD에서 변환된 데이터이므로 mllib 라이브러리를 사용한다.

```python
from pyspark.sql.functions import udf
#from pyspark.ml.linalg import DenseVector, VectorUDT
from pyspark.mllib.linalg import DenseVector, VectorUDT
#myudf=udf(lambda x: Vectors.dense(x), VectorUDT())
#myudf=udf(lambda x: Vectors.dense(x))
myudf=udf(lambda x: DenseVector(x.toArray()), VectorUDT())
_trainDf2=trainDf.withColumn('dvf',myudf(trainDf.features))
```

# RDD 훈련 데이터 만들기             
### 문제                   
머신러닝은 사람이 경험을 통해 배우는 것과 비슷하게 과거 데이터로부터 학습을 한다.                      
학습이란 어렵게 생각할 필요 없이, 과거 데이터에서 수학적이나 알고리즘을 활용하여 어떤 패턴을 찾아내는 것이다.                 
spark에서 제공한 데이터 파일을 읽어서 훈련 데이터를 만들어보자.             


### 해결                    
데이터를 읽어 RDD를 생성하고, label, features를 구성하여 Labeled Point로 만든다.         

### Python으로 파일 읽기
파일로부터 데이터를 읽기 위해, 파일명을 구성하고 try except 구문으로 입출력 오류를 확인할 수 있다.      

```python
import os

try:
  _fp = os.path.join(os.environ["SPARK_HOME"], \
    'data', 'mllib', 'sample_svm_data.txt')
except:
  print("An exception occurred")
```
파일로부터 데이터를 readlines() 함수로 모두 읽어온다.          
첫 행을 읽으면 label, features로 구성되어 있다.         

```python
_f = open(_fp, 'r')
_lines = _f.readlines()
_f.close()

print(_lines[0])
```

### Spark에서 RDD 생성
원본 데이터 sample_svm_data.txt는 공백으로 구분되어 있다.     
읽을 대상이 파일이므로, RDD를 사용한다.       
각 행을 공백으로 분리하여 읽는다.       
```python
_rdd = spark.sparkContext.textFile(_fp)\
  .map(lambda line : [float(x) for x in line.split(' ')])
```
각 행으로 분리되므로 2차원 리스트가 생성이 된다.       
첫째 행을 읽으려면 인덱스를 사용해야 한다.      
```python
_rdd.take(2)[0]
```

### Labeled Point 생성
위 데이터에서 보듯이 첫 열은 label로, 그 나머지는 features로 생성한다.      
```python
from pyspark.mllib.regression import LabeledPoint
_trainRDD0 = _rdd.map(lambda line: LabeledPoint(line[0], line[1:]))

_trainRDD0.take(1)
```
공백을 분리하고, 분리된 데이터를 labeled Point로 구성하는 기능을 합쳐서 실행해본다.           

```python
_trainRdd = spark.sparkContext.textFile(_fp)\
  .map(lambda line : [float(x) for x in line.split(' ')])\
  .map(lambda p : LabeledPoint(p[0], p[1:]))

_trainRdd.take(1)
```

### 정리하면
```python
def createLP(line):
  p = [float(x) for x in line.split(' ')]
  return LabeledPoint(p[0], p[1:])

_rdd = spark.sparkContext.textFile(_fp)
trainRdd = _rdd.map(createLP)

trainRdd.take(1)
```











