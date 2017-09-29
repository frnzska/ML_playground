"""Test file"""

from pyspark.sql import SparkSession
from pyspark.sql import SQLContext

spark = SparkSession.builder \
    .master("yarn-client") \
    .getOrCreate()

sc = spark.sparkContext

sqlContext = SQLContext(sc)
df = sqlContext.createDataFrame([(1, 'Carleen', 24, 245, 5),
                            (2, 'Steve', 31, 567, 7),
                            (3, 'Ann', 41, 354, 5),
                            (4, 'Lars', 30, 156, 3)], ('id', 'name', 'age', 'points', 'level'))
print(df.count())
df.show()

spark.stop()