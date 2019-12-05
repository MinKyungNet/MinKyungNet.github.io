---
layout: post
title: "trans vector"
tags: [sparse, dense, labeledpoint]
categories: [Big data analsis]
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

# 3.1 Vectors
```python
import numpy as np
dv = np.array([1.0, 2.1, 3])

from pyspark.mllib.linalg import Vectors
dv1mllib = Vectors.dense([1.0, 2.1, 3])
print(dv1mllib, type(dv1mllib))

from pyspark.ml.linalg import Vectors
dv1ml = Vectors.dense([1.0, 2.1, 3])
print(dv1ml)

>>>

[1.0,2.1,3.0] <class 'pyspark.mllib.linalg.DenseVector'>
[1.0,2.1,3.0]
```

dense vectors은 numpy array와 같은 특징을 가진다.      
인덱스로 값을 읽을 수 있다. 또한 반복문에서 사용할 수 있다.
```python
for e in dv1mllib:
    print(e)
    
print(dv1ml.dot(dv1ml))

print(np.dot(dv, dv))

print(dv1ml * dv1ml)

>>>

1.0
2.1
3.0
14.41
14.41
[1.0,4.41,9.0]
```

# 3.2 Sparse Vectors
1 차원 sparse vectors
```python
sv1 = Vectors.sparse(3, [1, 2], [1.0, 3.0])
print(sv1)
print(sv1.toArray())

>>>

(3,[1,2],[1.0,3.0])
[0. 1. 3.]
```

sparse vectors의 배열 방식 표현
```python
import numpy as np
import scipy.sparse as sps

row = np.array([0, 0, 1, 2, 2, 2])
col = np.array([0, 2, 2, 0, 1, 2])
data = np.array([1, 2, 3, 4, 5, 6])
mtx = sps.csc_matrix((data, (row, col)), shape=(3, 3))
print(mtx.todense())
print()
print(mtx)

>>>

[[1 0 2]
 [0 0 3]
 [4 5 6]]

  (0, 0)	1
  (2, 0)	4
  (2, 1)	5
  (0, 2)	2
  (1, 2)	3
  (2, 2)	6
```

Sparse Vectors의 CSR(Compressed Sparse Row)
```python
from pyspark.mllib.linalg import Matrix, Matrices
dm = Matrices.dense(6,4,[1, 2, 0, 0, 0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 5, 6, 7, 0, 0, 0, 0, 0, 0, 8])
dm.toArray()

>>>

array([[1., 0., 0., 0.],
       [2., 3., 0., 0.],
       [0., 0., 5., 0.],
       [0., 4., 6., 0.],
       [0., 0., 7., 0.],
       [0., 0., 0., 8.]])
```

```python
sm = Matrices.sparse(3, 2, [0, 1, 3], [0, 2, 1], [9, 6, 8])
d=sm.toDense()
print(d)

>>>

DenseMatrix([[9., 0.],
             [0., 8.],
             [0., 6.]])
```

RowMatrix
```python
p = [[1.0,2.0,3.0],[1.1,2.1,3.1],[1.2,2.2,3.3]]
my=spark.sparkContext.parallelize(p)
my.collect()
from pyspark.mllib.linalg.distributed import RowMatrix
rm=RowMatrix(my)
print(type(rm))
rm.rows.collect()

>>>

<class 'pyspark.mllib.linalg.distributed.RowMatrix'>
[DenseVector([1.0, 2.0, 3.0]),
 DenseVector([1.1, 2.1, 3.1]),
 DenseVector([1.2, 2.2, 3.3])]
```

3.3 Labeled Point
```python
from pyspark.mllib.regression import LabeledPoint
print(LabeledPoint(1.0, [1.0, 2.0, 3.0]))

>>>

(1.0,[1.0,2.0,3.0])
```

```python
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors
print(LabeledPoint(1992, Vectors.sparse(10, {0: 3.0, 1:5.5, 2: 10.0})))

>>>

(1992.0,(10,[0,1,2],[3.0,5.5,10.0]))
```

```python
from pyspark.mllib.regression import LabeledPoint
LabeledPoint(1.0, dv1mllib)

>>>

LabeledPoint(1.0, [1.0,2.1,3.0])
```

```python
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors
LabeledPoint(1.0, Vectors.fromML(dv1ml))

>>>

LabeledPoint(1.0, [1.0,2.1,3.0])
```

DataFrame에서 Labeled Point
```python
p = [[1,[1.0,2.0,3.0]],[1,[1.1,2.1,3.1]],[0,[1.2,2.2,3.3]]]
trainDf = spark.createDataFrame(p)
trainDf.collect()

>>>

[Row(_1=1, _2=[1.0, 2.0, 3.0]),
 Row(_1=1, _2=[1.1, 2.1, 3.1]),
 Row(_1=0, _2=[1.2, 2.2, 3.3])]
```


```python
from pyspark.mllib.regression import LabeledPoint
p = [LabeledPoint(1, [1.0, 2.0, 3.0]),
     LabeledPoint(1, [1.1, 2.1, 3.1]),
     LabeledPoint(0, [1.2, 2.2, 3.3])]
trainDf = spark.createDataFrame(p)
trainDf.collect()

>>>

[Row(features=DenseVector([1.0, 2.0, 3.0]), label=1.0),
 Row(features=DenseVector([1.1, 2.1, 3.1]), label=1.0),
 Row(features=DenseVector([1.2, 2.2, 3.3]), label=0.0)]
```

Vectors을 사용하여 생성
```python
from pyspark.mllib.linalg import Vectors
data = [(1.0, Vectors.dense([0.0, 1.1, 0.1])),
        (0.0, Vectors.dense([2.0, 1.0, 1.0])),
        (0.0, Vectors.dense([2.0, 1.3, 1.0])),
        (1.0, Vectors.dense([0.0, 1.2, 0.5]))]
trainDf = spark.createDataFrame(data, ["label", "features"])
trainDf.collect()

>>>

[Row(label=1.0, features=DenseVector([0.0, 1.1, 0.1])),
 Row(label=0.0, features=DenseVector([2.0, 1.0, 1.0])),
 Row(label=0.0, features=DenseVector([2.0, 1.3, 1.0])),
 Row(label=1.0, features=DenseVector([0.0, 1.2, 0.5]))]
```

schema를 사용하여 생성
```python
from pyspark.mllib.linalg import SparseVector, VectorUDT
from pyspark.sql.types import StructType, StructField, DoubleType
_rdd = spark.sparkContext.parallelize([
    (0.0, SparseVector(4, {1:1.0, 3:5.5})),
    (1.0, SparseVector(4, {0:-1.0, 2:0.5}))])

schema = StructType([
    StructField("label", DoubleType(), True),
    StructField("features", VectorUDT(), True)])

trainDf = _rdd.toDF(schema)
trainDf.collect()

>>>

[Row(label=0.0, features=SparseVector(4, {1: 1.0, 3: 5.5})),
 Row(label=1.0, features=SparseVector(4, {0: -1.0, 2: 0.5}))]
```

sparse에서 dense vector로 변환
```python
from pyspark.sql.functions import udf
from pyspark.mllib.linalg import DenseVector, VectorUDT
myudf = udf(lambda x : DenseVector(x.toArray()), VectorUDT())
_trainDf2 = trainDf.withColumn('dvf', myudf(trainDf.features))
_trainDf2.printSchema()
_trainDf2.show()

>>>

root
 |-- label: double (nullable = true)
 |-- features: vector (nullable = true)
 |-- dvf: vector (nullable = true)

+-----+--------------------+------------------+
|label|            features|               dvf|
+-----+--------------------+------------------+
|  0.0| (4,[1,3],[1.0,5.5])| [0.0,1.0,0.0,5.5]|
|  1.0|(4,[0,2],[-1.0,0.5])|[-1.0,0.0,0.5,0.0]|
+-----+--------------------+------------------+
```

# 문제 1 RDD 훈련 데이터 만들기
파일 읽기
```python
import os
try:
    _fp=os.path.join(os.environ["SPARK_HOME"], \
                    'data', 'mllib', 'sample_svm_data.txt')
except:
    print("An exception occurred")

_f = open(_fp,'r')
_lines = _f.readlines()
_f.close()

print(_lines[0])
>>>

1 0 2.52078447201548 0 0 0 2.004684436494304 2.000347299268466 0 2.228387042742021 2.228387042742023 0 0 0 0 0 0
```

spark에서 RDD 생성
```python
_rdd = spark.sparkContext.textFile(_fp)\
    .map(lambda line:[float(x) for x in line.split(' ')])

_rdd.take(1)[0]

>>>

[1.0,
 0.0,
 2.52078447201548,
 0.0,
 0.0,
 0.0,
 2.004684436494304,
 2.000347299268466,
 0.0,
 2.228387042742021,
 2.228387042742023,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0]
```

LabeledPoint 생성
```python
from pyspark.mllib.regression import LabeledPoint
_trainRdd0 = _rdd.map(lambda line : LabeledPoint(line[0], line[1:]))
_trainRdd0.take(1)

>>>

[LabeledPoint(1.0, [0.0,2.52078447201548,0.0,0.0,0.0,2.004684436494304,2.000347299268466,0.0,2.228387042742021,2.228387042742023,0.0,0.0,0.0,0.0,0.0,0.0])]
```


```python
_trainRdd = spark.sparkContext.textFile(_fp)\
    .map(lambda line: [float(x) for x in line.split(' ')])\
    .map(lambda p : LabeledPoint(p[0], p[1:]))
_trainRdd.take(1)

>>>

[LabeledPoint(1.0, [0.0,2.52078447201548,0.0,0.0,0.0,2.004684436494304,2.000347299268466,0.0,2.228387042742021,2.228387042742023,0.0,0.0,0.0,0.0,0.0,0.0])]
```

정리하면
```python
def createLP(line):
    p = [float(x) for x in line.split(' ')]
    return LabeledPoint(p[0], p[1:])

_rdd = spark.sparkContext.textFile(_fp)
trainRdd = _rdd.map(createLP)

trainRdd.take(1)

>>>

[LabeledPoint(1.0, [0.0,2.52078447201548,0.0,0.0,0.0,2.004684436494304,2.000347299268466,0.0,2.228387042742021,2.228387042742023,0.0,0.0,0.0,0.0,0.0,0.0])]
```
