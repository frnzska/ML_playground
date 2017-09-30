import pytest
import pyspark
import logging
py4j = logging.getLogger('py4j')
py4j.addHandler(logging.NullHandler())
py4j.propagate = False


@pytest.fixture(scope='module')
def spark_context():
    conf = pyspark.SparkConf()
    conf.setMaster('local')
    sc = pyspark.SparkContext.getOrCreate(conf=conf)
    return sc


@pytest.fixture
def example_df(spark_context):
    df = pyspark.sql.SQLContext(spark_context).createDataFrame(
        [('2007-07-03 19:41:00', 1),
         ('2007-10-03 19:41:00', 2),
         ('2007-07-03 20:44:10', 3)],
         ('created_at', 'uuid'))
    return df

def test_method1(example_df):
    # do something with exmaple_df ...
    assert 1 == 1