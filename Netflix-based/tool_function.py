from matrix_factorization import MatrixFactorization
from scipy.sparse import find
import numpy as np

########################################## For Validation #############################################
def calculate_avg_rating_for_pesudo_user(pseudo_user_lst, sMatrix):
    '''ret_dict: dict {
        itemid0: rating0 
        itemid1: rating1
        ...             
    }'''

    ret_array = np.zeros(sMatrix.shape[0])
    ret_array = np.array(sMatrix[:, pseudo_user_lst].sum(axis=1))[:,0]/(sMatrix[:, pseudo_user_lst].getnnz(axis=1)+1e-9)

    return ret_array


def pred_RMSE_for_validate_user(user_node_ind, user_profile, item_profile, val_user_list, val_item_list, sMatrix):
    RMSE = 0
    for userid, itemid in zip(val_user_list, val_item_list):
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
    val_item_list = find(validation_set)[0]
    val_user_list = find(validation_set)[1]
    user_node_ind = np.zeros(sMatrix.shape[1])                  #### notice that index is not id
    
    for level in lr_bound:
        prediction_model.setdefault(level, {})
        train_lst = []       
        for pseudo_user_bound, userid in zip(lr_bound[level], range(len(lr_bound[level]))):
            if pseudo_user_bound[0] > pseudo_user_bound[1]:
                continue
            pseudo_user_lst = tree[pseudo_user_bound[0]:(pseudo_user_bound[1] + 1)]
            pseudo_user_for_item = calculate_avg_rating_for_pesudo_user(pseudo_user_lst, sMatrix)
            train_lst += [(userid, int(itemid), float(pseudo_user_for_item[itemid])) for itemid in range(pseudo_user_for_item.shape[0]) if pseudo_user_for_item[itemid]]    
            #### find node index for each validation user ####
            user_node_ind[pseudo_user_lst] = userid      

        #### Train MF and Do validation ####
        min_RMSE = -1
        for plambda in plambda_candidates[level]:
            MF.change_parameter(plambda)
            user_profile, item_profile = MF.matrix_factorization(train_lst)
            RMSE = pred_RMSE_for_validate_user(user_node_ind, user_profile, item_profile, val_user_list, val_item_list, sMatrix)
            if min_RMSE is -1 or RMSE < min_RMSE:
                min_RMSE = RMSE
                min_user_profile, min_item_profile, min_lambda = user_profile, item_profile, plambda
        prediction_model[level]['upro'], prediction_model[level]['ipro'], prediction_model[level]['plambda'] \
                                             = min_user_profile, min_item_profile, min_lambda
    MF.end()   #### close MF spark session
    return prediction_model
#######################################################################################################


############################################# For Test ################################################
def RMSE(real_rating, non_zeros_rating_cnt, pred_rating, rated_item):
    rmse1 = np.sum((pred_rating-real_rating)**2)
    rmse2 = np.sum((pred_rating[rated_item] - real_rating[rated_item])**2)
    return ((rmse1-rmse2)/(non_zeros_rating_cnt-len(rated_item)))**0.5
    # for itemid, rating in real_rating.items():
    #     if itemid not in rated_item:
    #         rmse += (pred_rating[itemid-1] - rating)**2
    #         cnt += 1
    # return (rmse/cnt)**0.5


def predict(user_profile, item_profile):
    ''' 
        user_profile: array {
                        [k1, k2, k3, ... , kt]
                    } profile for certain user
        item_profile: dict {
                        itemid1: [k1, k2, k3, ... , kt], 
                        itemid2: [k1, k2, k3, ... , kt], 
                        itemid3: [k1, k2, k3, ... , kt], 
                    } profile for items in each node
     '''
    # item_profile_cont = np.array(list(item_profile.values()))  # shape: (I, k)
    #### Calculate predict rating ####
    pred_rating = np.dot(item_profile, user_profile)
    # pred_rating = { itemid: np.dot(item_profile_cont[i], user_profile) \
    #                     for itemid, i in zip(item_profile, range(item_profile_cont.shape[0])) }
    return pred_rating

def pred_RMSE_for_new_user(split_item, rI, prediction_model, sM_testing):
    '''
        sM_testing: 30% test dataset (sparse matrix)
        split_item: list [
                level 0: [112],
                level 1: [48, 0, 79],
                level 2: [15, 0, 17, 1, 1, 1, 61, 0, 50]
                ...
            ]
        User: dict {
                    userid1: { itemid11: rating11, itemid12: rating12, ... } rating of user 1
                    userid2: { itemid21: rating21, itemid22: rating22, ... } rating of user 2
                    userid3: ...
                }
        return : rmse value (float)
    '''

    non_zeros_rating_cnt = sM_testing.getnnz(axis=0)
    rmse = 0
    for userid in range(sM_testing.shape[1]):
        pred_index = 0
        # new_user_ratings = []
        final_level = 0
        rated_item = []
        user_all_ratings = sM_testing[:,userid].nonzero()[0]
        for level in range(len(split_item)):
            if split_item[level][pred_index] not in user_all_ratings:
                tmp_pred_index = 3*pred_index + 2
                if tmp_pred_index in prediction_model[str(int(level)+1)]['upro']:
                    # new_user_ratings.append([split_item[level][pred_index], 0])
                    final_level += 1
                    pred_index = tmp_pred_index
                else:
                    break
            elif sM_testing[split_item[level][pred_index], userid] >= 4:
                tmp_pred_index = 3*pred_index
                if tmp_pred_index in prediction_model[str(int(level)+1)]['upro']:
                    rated_item.append(split_item[level][pred_index]-1)
                    # new_user_ratings.append([split_item[level][pred_index], sM_testing[split_item[level][pred_index], userid]])
                    final_level += 1
                    pred_index = tmp_pred_index
                else:
                    break
            elif sM_testing[split_item[level][pred_index], userid] <= 3:
                tmp_pred_index = 3*pred_index + 1
                if tmp_pred_index in prediction_model[str(int(level)+1)]['upro']:
                    rated_item.append(split_item[level][pred_index]-1)
                    # new_user_ratings.append([split_item[level][pred_index], sM_testing[split_item[level][pred_index], userid]])
                    final_level += 1
                    pred_index = tmp_pred_index
                else:
                    break
        pred_rating = predict(np.array(prediction_model[str(final_level)]['upro'][pred_index]), \
                                            np.array(list(prediction_model[str(final_level)]['ipro'].values())))
        rmse += RMSE(sM_testing[1:, userid].toarray(), non_zeros_rating_cnt[userid], pred_rating, rated_item)

    return rmse / sM_testing.shape[1]
#######################################################################################################