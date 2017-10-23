from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .master("yarn-client") \
    .getOrCreate()


def get_data(test_set='s3://franziska-adler-emr/data/mnist/mnist_test.csv',
             train_set='s3://franziska-adler-emr/data/mnist/mnist_train.csv'):
    test = spark.read.load(test_set, format('csv'), inferSchema=True)
    train = spark.read.load(train_set, format('csv'), inferSchema=True)
    return test, train

def prepare_data(test, train):
    # prepare vectors: Spark wants vectors called features
    train = train.withColumnRenamed("_c0", "label")
    test = test.withColumnRenamed("_c0", "label")
    feature_cols = ["_c" + str(i) for i in range(1, 784)]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    train = assembler.transform(train).select('label', 'features')
    test = assembler.transform(test).select('label', 'features')
    return test, train


dims_in = 783
dims_out = 10
n_hidden = 100
layers = [dims_in, n_hidden, dims_out]
maxIter = 100
learning_rate = 1e-7

test, train = get_data()
test, train = prepare_data(test, train)

trainer = MultilayerPerceptronClassifier(maxIter=maxIter, layers=layers, blockSize=128, seed=1234, tol=learning_rate)
model = trainer.fit(train)

result = model.transform(test)
predictionAndLabels = result.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))


spark.stop()

# ssh hadoop@xyzMaster
# /usr/lib/Spark/bin/Spark-submit