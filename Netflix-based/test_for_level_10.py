import klepto
import shelve
import pickle
import numpy as np

from scipy.sparse import *


from pyspark.mllib.recommendation import ALS
from pyspark.sql import SparkSession



############### Load Data ##################
rating_matrix_csc = load_npz('netflix/sparse_matrix_100%.npz').tocsc()
rating_matrix_val_csc = load_npz('netflix/sparse_matrix_validation_75%.npz').tocsc()
print("file load DONE")
############################################

''' Save to file 'tree.pkl' '''
start = 0
end = int(rating_matrix_csc.shape[1] * 0.75)

from pyspark.mllib.recommendation import ALS
from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark import SparkContext

class MatrixFactorization:
    def __init__(self, maxIter=15, regParam=0.01, rank=10):
        self.maxIter = maxIter
        self.regParam = regParam
        self.rank = rank
        conf = SparkConf().setAppName("appName").setMaster("local[*]")
        conf.set("spark.driver.memory","8g")
        conf.set("spark.executor.memory","8g")
        self.spark = SparkContext(conf=conf)

                    
                    
        print("New SparkSession started...")

    def change_parameter(self, regParam):
        self.regParam = regParam

    def matrix_factorization(self, train_lst):
        ratings = self.spark.parallelize(train_lst)
        print('create dataframe!')
        model = ALS.train(ratings, self.rank, seed=10, \
                          iterations=self.maxIter, \
                          lambda_=self.regParam)
        print("MF DONE")
        userFeatures = sorted(model.userFeatures().collect(), key=lambda d: d[0], reverse=False)
        productFeatures = sorted(model.productFeatures().collect(), key=lambda d: d[0], reverse=False)
        userProfile = {each[0]: each[1].tolist() for each in userFeatures}
        itemProfile = {each[0]: each[1].tolist() for each in productFeatures}
        
        
        return userProfile, itemProfile

    def end(self):
        self.spark.stop()
        print("SparkSession stopped.")



from scipy.sparse import find
val_num = rating_matrix_val_csc.getnnz(axis=None)
########################################## For Validation #############################################
def calculate_avg_rating_for_pesudo_user(pseudo_user_lst, sMatrix):

    ret_array = np.zeros(sMatrix.shape[0])
    ret_array = np.array(sMatrix[:, pseudo_user_lst].sum(axis=1))[:,0]/(sMatrix[:, pseudo_user_lst].getnnz(axis=1)+1e-9)

    return ret_array


def pred_RMSE_for_validate_user(user_node_ind, user_profile, item_profile, val_user_list, val_item_list, sMatrix):
    print("RMSE calculation on valset started.")
    RMSE = 0
    i = 0
    for userid, itemid in zip(val_user_list, val_item_list):
        if i % 100000 == 0:
            print("%.2f%%" % (100 * i / val_num))        
        i += 1
        RMSE += (sMatrix[itemid, userid] - np.dot(user_profile[user_node_ind[userid]], item_profile[itemid]))**2
    return (RMSE / len(val_user_list))**0.5

def generate_prediction_model(lr_bound, tree, rI, sMatrix, plambda_candidates, validation_set):
    ''' lr_bound: dict {
                level 0: [[left_bound, right_bound]], users' bound for one level, each ele in dictionary represents one node
                level 1: [[left_bound, right_bound], [left_bound, right_bound], [left_bound, right_bound]], 3
                level 2: ..., 9
            } (bound means index)
        plambda_candidates: {
            level 0: [clambda1, clambda2, clambda3, ...]
            level 1: [clambda1, clambda2, clambda3, ...]
            level 2: [clambda1, clambda2, clambda3, ...]
        }
        prediction_model: dict {
                level 0: { 'best_lambda': x, 'user_profile': ..., 'item_profile': ...}
                level 1: { 'best_lambda': x, 'user_profile': ..., 'item_profile': ...}
                level 2: { 'best_lambda': x, 'user_profile': ..., 'item_profile': ...}
            }
    '''
    MF = MatrixFactorization()
    prediction_model = {}
    rmst_dict = {"0":[],
                     "1":[],
                     "2":[],
                     "3":[],
                     "4":[],
                     "5":[],
                     "6":[],
                     "7":[],
                     "8":[],
                     "9":[],
                     "10":[]}
    
    val_item_list = find(validation_set)[0]
    val_user_list = find(validation_set)[1]
    user_node_ind = np.zeros(sMatrix.shape[1])                  #### notice that index is not id
    
    for level in lr_bound:
        if int(level) <= 5:
            continue 
        print("level:", level)
        prediction_model.setdefault(level, {})
        train_lst = []       
        for pseudo_user_bound, userid in zip(lr_bound[level], range(len(lr_bound[level]))):
#             print(str(userid) + "/" + str(pow(3,int(level))))
            if pseudo_user_bound[0] > pseudo_user_bound[1]:
                continue
            pseudo_user_lst = tree[pseudo_user_bound[0]:(pseudo_user_bound[1] + 1)]
            pseudo_user_for_item = calculate_avg_rating_for_pesudo_user(pseudo_user_lst, sMatrix)
            train_lst += [(userid, itemid, float(pseudo_user_for_item[itemid])) \
                          for itemid in range(pseudo_user_for_item.shape[0]) if pseudo_user_for_item[itemid]]    
            #### find node index for each validation user ####
            user_node_ind[pseudo_user_lst] = userid      

        print("Rating Number of level " + level + ": " + str(len(train_lst)))
        #### Train MF and Do validation ####
        min_RMSE = -1
        for plambda in plambda_candidates[level]:
        # MF = MatrixFactorization()
            print(plambda)
            MF.change_parameter(plambda)
            user_profile, item_profile = MF.matrix_factorization(train_lst)
            # MF.end()   #### close MF spark session
            # del MF
            RMSE = pred_RMSE_for_validate_user(user_node_ind, user_profile, item_profile, val_user_list, val_item_list, validation_set)
            rmst_dict[level].append(RMSE)
            if min_RMSE is -1 or RMSE < min_RMSE:
                min_RMSE = RMSE
                min_user_profile, min_item_profile, min_lambda = user_profile, item_profile, plambda_candidates[level]
                
        print("min_lambda: " + str(min_lambda))
        print("min_RMSE: " + str(min_RMSE))

        import matplotlib.pyplot as plt

        plt.figure(1)
        plt.title('avg RMSE for each iteration')
        plt.xlabel('theta')
        plt.ylabel('avg RMSE')
        plt.plot(plambda_candidates[level], rmst_dict[level])
        # plt.plot(param_list, rmse_list)
        plt.show()

        prediction_model[level]['upro'], prediction_model[level]['ipro'], prediction_model[level]['plambda'] \
                                             = min_user_profile, min_item_profile, min_lambda
        print("level " + level + " training DONE")
    
    MF.end()
    return prediction_model

import klepto
import numpy as np
Tree = klepto.archives.dir_archive('treeFile', {}, serialized=True)
Tree.load()

plambda_candidates = {"0":[0.001, 0.0011, 0.0012, 0.0013, 0.0014, 0.0015, 0.0016, 0.0017, 0.0018, 0.0019],
                     "1":[0.001, 0.0011, 0.0012, 0.0013, 0.0014, 0.0015, 0.0016, 0.0017, 0.0018, 0.0019],
                     "2":[0.001, 0.0011, 0.0012, 0.0013, 0.0014, 0.0015, 0.0016, 0.0017, 0.0018, 0.0019],
                     "3":[0.001, 0.0011, 0.0012, 0.0013, 0.0014, 0.0015, 0.0016, 0.0017, 0.0018, 0.0019],
                     "4":[0.0021, 0.0022, 0.0023, 0.0024, 0.0025, 0.0026, 0.0027, 0.0028, 0.0029, 0.030],
                     "5":[0.0021, 0.0022, 0.0023, 0.0024, 0.0025, 0.0026, 0.0027, 0.0028, 0.0029, 0.030],
                     "6":[0.009, 0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019],
                     "7":[0.009, 0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019],
                     "8":[0.018, 0.019, 0.02, 0.021, 0.022, 0.023, 0.024, 0.025, 0.026, 0.027, 0.028],
                     "9":[0.018, 0.019, 0.02, 0.021, 0.022, 0.023, 0.024, 0.025, 0.026, 0.027, 0.028],
                     "10":[0.018, 0.019, 0.02, 0.021, 0.022, 0.023, 0.024, 0.025, 0.026, 0.027, 0.028]}
# for level in Tree["lr_bound"]:
#     plambda_candidates[level] = list(np.arange(0.001, 0.05, 0.005))    


prediction_model = generate_prediction_model \
            (Tree['lr_bound'], \
             Tree['tree'], \
             Tree['rI'], \
             rating_matrix_csc[:, start:end], 
             plambda_candidates, 
             rating_matrix_val_csc)

import pickle
import shelve

d = shelve.open("prediction_model", protocol=pickle.HIGHEST_PROTOCOL)
d["content"] = prediction_model
d.close()
print("Write DONE!")