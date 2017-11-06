import pandas as pd
from scipy.sparse import *
from scipy import *
import tool_function as tf

######################### Step 1: Input Dataset #########################
catagory = "Cell_Phones_and_Accessories"
item_data = 'item_metadata/meta_' + catagory + '.csv'
rating_data = 'user_ratings/' + catagory + '.csv'
# print(rating_data)
ratingsFrame = pd.read_csv(rating_data, names = ["userID", "itemID", "rating"])
ratingsFrame.sort_values(by = 'userID', ascending=True, inplace=True)
itemsFrame = pd.read_csv(item_data)
#########################################################################

############### Step 2: Construct User-Item Sparse Matrix ###############
item_lst = ratingsFrame["itemID"].tolist()
user_lst = ratingsFrame["userID"].tolist()
rating_lst = ratingsFrame["rating"].tolist()
row = []
col = []
rating_data = []
for i in range(ratingsFrame.shape[0]):
    row.append(item_lst[i])
    col.append(user_lst[i])
    rating_data.append(rating_lst[i])
# print(user_lst[len(user_lst)-1])
rating_martix_coo = coo_matrix((rating_data, (row, col)), shape=(itemsFrame.shape[0], user_lst[len(user_lst)-1]+1))
rating_martix_csc = rating_martix_coo.tocsc()
rating_martix_csr = rating_martix_coo.tocsr()
#########################################################################

################## Step 3: Split dataset into training and test dataset ##################
start = 0
end = int(rating_martix_csc.shape[1] * 0.7)
sM_training = rating_martix_csc[:, start:end]
sM_testing =  rating_martix_csc[:, end:]
##########################################################################################

############## Step 4: Initialize FDT tree and construct prediction model ################
dtmodel_realdata = DecisionTreeModel(sM_training, depth_threshold=3)
dtmodel_realdata.build_model()
##########################################################################################

######################## Step 5: Calculate RMSE on test dataset ##########################
RMSE = tf.pred_RMSE_for_new_item(dtmodel_realdata, sM_testing)
##########################################################################################

################################### Step 6: stop spark ###################################
dtmodel_realdata.MF.end()
##########################################################################################