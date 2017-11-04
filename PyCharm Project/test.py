import pandas as pd

catagory = "Grocery_and_Gourmet_Food"
item_data = '../item_metadata/meta_' + catagory + '.csv'
rating_data = '../user_ratings/' + catagory + '.csv'

ratingsFrame = pd.read_csv(rating_data, names = ["userID", "itemID", "rating"])
ratingsFrame.sort_values(by = 'itemID', ascending=True, inplace=True)
rating_train = ratingsFrame.to_records(index=False).tolist()
# ratingsFrame
# rating_test = ratingsFrame[int(ratingsFrame.shape[0]*0.7):].to_records(index=False).tolist()

from pyspark.mllib.recommendation import ALS
from pyspark.sql import SparkSession


class MatrixFactorization:
    def __init__(self, maxIter=15, regParam=0.01, rank=10):
        self.maxIter = maxIter
        self.regParam = regParam
        self.rank = rank
        self.spark = SparkSession \
            .builder \
            .appName("Python Spark SQL basic example") \
            .config("spark.some.config.option", "some-value") \
            .getOrCreate()

    def matrix_factorization(self, train_lst):
        ratings = self.spark.createDataFrame(train_lst)
        model = ALS.train(ratings, self.rank, seed=10, \
                          iterations=self.maxIter, \
                          lambda_=self.regParam)
        print("MF DONE")
        userFeatures = sorted(model.userFeatures().collect(), key=lambda d: d[0], reverse=False)
        productFeatures = sorted(model.productFeatures().collect(), key=lambda d: d[0], reverse=False)
        userProfile = [each[1].tolist() for each in userFeatures]
        itemProfile = [each[1].tolist() for each in productFeatures]

        return userProfile, itemProfile

    def end(self):
        self.spark.stop()

mf = MatrixFactorization()
print("MF DONE")
for i in range(5):
    rating_train = [(0, 0, 2.9999999992499999), (0, 1, 2.6666666657777776), (0, 2, 2.9999999989999999), (0, 3, 1.9999999979999998), (0, 4, 3.3333333322222223)]
    userProfile, itemProfile = mf.matrix_factorization(rating_train)

mf.end()





import pickle
import numpy as np
output = open('data.pkl', 'wb')
string1 = "hello1"
string2 = [1,2,3]
string3 = {'a':1}
string4 = np.array([[1,2,3],[4,5,6]])
pickle.dump(string1, output)
pickle.dump(string2, output)
pickle.dump(string3, output)
pickle.dump(string4, output)
output.close()

inputf = open('data.pkl', 'rb')
s1 = pickle.load(inputf)
s2 = pickle.load(inputf)
s3 = pickle.load(inputf)
s4 = pickle.load(inputf)
print(s1)
print(s2)
print(s3)
print(s4)
output.close()