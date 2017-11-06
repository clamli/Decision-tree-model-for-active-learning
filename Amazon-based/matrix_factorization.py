from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS
from pyspark.sql import SparkSession

class MatrixFactorization:
    def __init__(self, maxIter = 15, regParam = 0.01, rank = 10):
        self.maxIter = maxIter
        self.regParam = regParam  
        self.rank = rank  
        self.spark = SparkSession.builder.master("local[*]").appName("Example").getOrCreate()
        
    def matrix_factorization(self, train_lst):
        
        ratings = self.spark.createDataFrame(train_lst)
        model = ALS.train(ratings, self.rank, seed=10, \
                          iterations = self.maxIter, \
                          lambda_ = self.regParam)
        print("MF DONE")
        userFeatures = sorted(model.userFeatures().collect(), key=lambda d:d[0], reverse = False)
        productFeatures = sorted(model.productFeatures().collect(), key=lambda d:d[0], reverse = False)
        userProfile = {each[0]: each[1].tolist() for each in userFeatures}
        itemProfile = {each[0]: each[1].tolist() for each in productFeatures}
            
        return userProfile, itemProfile

    def end(self):
    	self.spark.stop()