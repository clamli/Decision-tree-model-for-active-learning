import numpy as np
from scipy.sparse import find
from copy import deepcopy
from matrix_factorization import MatrixFactorization

class DecisionTreeModel:
    def __init__(self, sMatrix, depth_threshold=6, plambda=7, MSP_item=200):
        '''
            sMatrix: I*U matrix
            depth_threshold: terminate depth
            plambda: regularization parameter
            self.rI: dict { 
                        itemid1: [ [uid11, rating11], [uid12, rating12], ... ] rating for item 1
                        itemid2: [ [uid21, rating21], [uid22, rating22], ... ] rating for item 2
                        itemid3: ...
                     }
            self.rU: dict {
                        userid1: { itemid11: rating11, itemid12: rating12, ... } rating of user 1
                        userid2: { itemid21: rating21, itemid22: rating22, ... } rating of user 2
                        userid3: ...
                     }
            self.lr_bound: dict {
                                level 0: [[left_bound, right_bound]], users' bound for one level, each ele in dictionary represents one node
                                level 1: [[left_bound, right_bound], [left_bound, right_bound], [left_bound, right_bound]], 3
                                level 2: ..., 9
                            } (bound means index)
            self.tree: []  all of userid
            self.split_item: list [
                    level 0: []
                    level 1: []
                    level 2: []
            ]
            self.sum_cur_t: dict {
                        itemid1: {'rating': sum of ratings for item 1, 'cnt': sum of users rated item 1}
                        itemid2: {'rating': sum of ratings for item 2, 'cnt': sum of users rated item 2}
                        ...
                    }
            self.sum_2_cur_t: dict {
                        itemid1: sum of square ratings for item 1
                        itemid2: sum of square ratings for item 2
                        ...
                    }
            self.biasU: dict {
                        userid1: bias1
                        userid2: bias2
                        ...
                    }
            self.user_profile: dict {
                        level 0: {pseudo_user1: [k1, k2, k3, ... , kt]}
                        level 1: {pseudo_user1: [k1, k2, k3, ... , kt], pseudo_user2: [k1, k2, k3, ... , kt], pseudo_user3: [k1, k2, k3, ... , kt]}
                        ... 
                    } profile for each level's node
            self.item_profile: dict {
                        level 0: {itemid1: [k1, k2, k3, ... , kt], itemid2: [k1, k2, k3, ... , kt], itemid3: [k1, k2, k3, ... , kt], ...} for each item
                        level 1: {itemid1: [k1, k2, k3, ... , kt], itemid2: [k1, k2, k3, ... , kt], itemid3: [k1, k2, k3, ... , kt], ...} for each item
                        ...
                    } profile for each item
            every element represents ratings for one item, its order decide the users in tree nodes
        '''
        self.depth_threshold = depth_threshold
        self.plambda = plambda
        self.cur_depth = 0
        self.MSP_item = MSP_item
        self.real_item_num = sMatrix.shape[0]
        x = find(sMatrix)
        itemset = x[0]
        userset = x[1]	
        # self.rI = {}
        self.rU = {}	
        self.sum_cur_t = {}
        self.sum_2_cur_t = {}
        # self.rI[itemset[0]] = [[userset[0], sMatrix[itemset[0], userset[0]]]]
        # self.rU[userset[0]] = {itemset[0]: sMatrix[itemset[0], userset[0]]}
        self.global_mean = 0                   # global average of ratings

        #### Calculate rate of progress ####
        self.node_num = 0
        self.cur_node = 0
        for i in range(self.depth_threshold):
            self.node_num += 3**i

        #### Generate rI, rU ####
        self.rI = list(set(sMatrix.nonzero()[0]))
        for itemid, userid in zip(itemset, userset):
            self.rU.setdefault(userid, {})[itemid] = sMatrix[itemid, userid]
            # self.rI.setdefault(itemid, []).append([userid, sMatrix[itemid, userid]])
            self.global_mean += sMatrix[itemid, userid]
        self.global_mean /= len(itemset)
        self.item_size = len(self.rI)
        self.user_size = len(self.rU)

        #### Initiate Tree, lr_bound ####
        self.tree = list(self.rU.keys())
        self.split_item = []
        self.lr_bound = {'0': [[0, len(self.tree)-1]]}

        #### Generate bias, sum_cur_t, sum_2_cur_t ####
        self.biasU = {}
        self.sum_cur_t = np.zeros(self.real_item_num)
        self.sum_2_cur_t = np.zeros(self.real_item_num)
        self.sum_cntt = np.zeros(self.real_item_num)
#         self.sum_cur_t[itemset[0]] = {'rating': sMatrix[itemset[0], userset[0]]-self.biasU[userset[0]], 'cnt': 1}
#         self.sum_2_cur_t[itemset[0]] = (sMatrix[itemset[0], userset[0]]-self.biasU[userset[0]])**2
        for userid in self.rU:
            self.biasU[userid] = (sum(list(self.rU[userid].values())) + self.plambda*self.global_mean) / (self.plambda + len(self.rU[userid]))
            user_all_rating_id = np.array(list(self.rU[userid].keys()))
            # print('user_all_rating_id ', user_all_rating_id[:])
            user_all_rating = np.array(list(self.rU[userid].values()))
            self.sum_cur_t[user_all_rating_id[:]] += user_all_rating[:] - self.biasU[userid]
            self.sum_2_cur_t[user_all_rating_id[:]] += (user_all_rating[:] - self.biasU[userid])**2
            self.sum_cntt[user_all_rating_id[:]] += 1
        
#         for itemid, userid, ind in zip(itemset[1:],userset[1:],range(1, len(itemset))):
#             if itemid == itemset[ind-1]:
#                 self.sum_cur_t[itemid]['rating'] += sMatrix[itemid, userid]-self.biasU[userid]
#                 self.sum_cur_t[itemid]['cnt'] += 1
#                 self.sum_2_cur_t[itemid] += (sMatrix[itemid, userid]-self.biasU[userid])**2
#             else:
#                 self.sum_cur_t[itemid] = {'rating': sMatrix[itemid, userid]-self.biasU[userid], 'cnt': 1}
#                 self.sum_2_cur_t[itemid] = (sMatrix[itemid, userid]-self.biasU[userid])**2

        #### Prediction Model ####
        self.user_profile = {}
        self.item_profile = {}
        self.MF = MatrixFactorization()

        print("Initiation DONE!")


    def calculate_error(self, sumt, sumt_2, cntt):
        ''' Calculate error for one item-split in one node '''
        Error_i = np.sum(sumt_2 - (sumt**2)/(cntt+1e-9))
#         for itemid in sumtL:
#             Error_i += sumtL_2[itemid] - (sumtL[itemid]['rating']**2)/(sumtL[itemid]['cnt']+1e-9) \
#                         + sumtD_2[itemid] - (sumtD[itemid]['rating']**2)/(sumtD[itemid]['cnt']+1e-9) \
#                             + sumtU_2[itemid] - (sumtU[itemid]['rating']**2)/(sumtU[itemid]['cnt']+1e-9)
        return Error_i


    def generate_decision_tree(self, lr_bound_for_node, chosen_id):
        '''
            sumtL: dict {
                itemid1: {'rating': sum of ratings for item 1, 'cnt': sum of users rated item 1}
                itemid2: {'rating': sum of ratings for item 2, 'cnt': sum of users rated item 2}
                ...
            }
            sumtL_2: dict {
                itemid1: sum of square ratings for item 1
                itemid2: sum of square ratings for item 2
                ...
            }
            lr_bound_for_node: list [leftind, rightind] for one node
        '''

        #### Terminate ####
        self.cur_depth += 1
        if self.cur_depth > self.depth_threshold or len(chosen_id) == self.item_size:
            return
        
        #### Choose Most Popular Items of This Node ####
        num_rec = np.zeros(self.real_item_num)      
        for userid in self.tree[lr_bound_for_node[0]:(lr_bound_for_node[1]+1)]:
            user_all_rating_id = np.array(list(self.rU[userid].keys()))
            num_rec[user_all_rating_id[:]] += 1
        sub_item_id = np.argsort(num_rec)[:self.MSP_item]
        

        #### Find optimum item to split ####
        min_sumtL, min_sumtD, min_sumtL_2, min_sumtD_2, min_sumtU, min_sumtU_2, Error = {}, {}, {}, {}, {}, {}, {}
        min_Error = "None"
        for itemid in sub_item_id:
            if itemid in chosen_id:
                continue
            ''' 
                user_rating_item_in_nodet: [ [uid01, rating01], [uid02, rating02], ... ] 
                to find all users in node t who rates item i
            '''
            user_rating_item_in_nodet = ([userid, self.rU[userid][itemid]] for userid in self.tree[lr_bound_for_node[0]:(lr_bound_for_node[1]+1)] if itemid in self.rU[userid])
            sumt = np.zeros((self.real_item_num, 3))
            sumt_2 = np.zeros((self.real_item_num, 3))
            cntt = np.zeros((self.real_item_num, 3))
            for user in user_rating_item_in_nodet:
                ''' user_all_rating: array [ [itemid11, rating11], [itemid12, rating12], ... ] '''
                user_all_rating_id = np.array(list(self.rU[user[0]].keys()))
                user_all_rating = np.array(list(self.rU[user[0]].values()))
                #### calculate sumtL for node LIKE ####
                if user[1] >= 4:
                    sumt[user_all_rating_id[:], 0] += user_all_rating[:] - self.biasU[user[0]]
                    sumt_2[user_all_rating_id[:], 0] += (user_all_rating[:] - self.biasU[user[0]])**2
                    cntt[user_all_rating_id[:], 0] += 1
                #### calculate sumtD for node DISLIKE ####
                elif user[1] <= 3:
                    sumt[user_all_rating_id[:], 1] += user_all_rating[:] - self.biasU[user[0]]
                    sumt_2[user_all_rating_id[:], 1] += (user_all_rating[:] - self.biasU[user[0]])**2
                    cntt[user_all_rating_id[:], 1] += 1
            #### calculate sumtU for node UNKNOWN ####
            sumt[:, 2] = self.sum_cur_t[:] - sumt[:, 0] - sumt[:, 1]
            sumt_2[:, 2] = self.sum_2_cur_t[:] - sumt_2[:, 0] - sumt_2[:, 1]
            cntt[:, 2] = self.sum_cntt[:] - cntt[:, 0] - cntt[:, 1]
            Error[itemid] = self.calculate_error(sumt, sumt_2, cntt)            
                
                
#             sumtL, sumtD, sumtL_2, sumtD_2, sumtU, sumtU_2 = {}, {}, {}, {}, {}, {}
#             sumtL = {k:{'rating': 0, 'cnt': 0} for k in self.rI.keys()}
#             sumtL_2 = sumtL_2.fromkeys(self.rI.keys(), 0)
#             sumtD = {k:{'rating': 0, 'cnt': 0} for k in self.rI.keys()}
#             sumtD_2 = sumtD_2.fromkeys(self.rI.keys(), 0)
#             for user in user_rating_item_in_nodet:
#                 ''' user_all_rating: [ [itemid11, rating11], [itemid12, rating12], ... ] '''
#                 user_all_rating = self.rU[user[0]]
#                 #### calculate sumtL for node LIKE ####
#                 if user[1] >= 4:
#                     for uritem, rating in user_all_rating.items():
#                         sumtL[uritem]['rating'] += rating
#                         sumtL_2[uritem] += (rating-self.biasU[user[0]])**2
#                         sumtL[uritem]['rating'] -= self.biasU[user[0]]
#                         sumtL[uritem]['cnt'] += 1
#                 #### calculate sumtD for node DISLIKE ####
#                 elif user[1] <= 3:
#                     for uritem, rating in user_all_rating.items():
#                         sumtD[uritem]['rating'] += rating
#                         sumtD_2[uritem] += (rating-self.biasU[user[0]])**2
#                         sumtD[uritem]['rating'] -= self.biasU[user[0]]
#                         sumtD[uritem]['cnt'] += 1
#             #### calculate sumtU for node UNKNOWN ####
#             for iid in self.rI:
#                 sumtU[iid] = {}
#                 sumtU[iid]['rating'] = self.sum_cur_t[iid]['rating'] - sumtL[iid]['rating'] - sumtD[iid]['rating']
#                 sumtU[iid]['cnt'] = self.sum_cur_t[iid]['cnt'] - sumtL[iid]['cnt'] - sumtD[iid]['cnt']
#                 sumtU_2[iid] = self.sum_2_cur_t[iid] - sumtL_2[iid] - sumtD_2[iid]
#             #### calculate error by (eL + eD + eU) ####
#             Error[itemid] = self.calculate_error(sumtL, sumtL_2, sumtD, sumtD_2, sumtU, sumtU_2)
            if min_Error == "None" or Error[itemid] < min_Error:
                min_sumt = sumt
                min_sumt_2 = sumt_2
                min_cntt = cntt
                min_Error = Error[itemid]
        #### Find optimum split-item ####
        optimum_itemid = min(Error, key=Error.get)
        if len(self.split_item) == self.cur_depth-1:
            self.split_item.append([optimum_itemid])
        else:
            self.split_item[self.cur_depth-1].append(optimum_itemid)
        # self.split_item.setdefault(str(self.cur_depth-1), []).append(optimum_itemid)
        chosen_id.append(optimum_itemid)


        #### sort tree ####
        self.lr_bound.setdefault(str(self.cur_depth), []).append([])                                          # for LIKE
        self.lr_bound[str(self.cur_depth)].append([])   					                                  # for DISLIKE
        self.lr_bound[str(self.cur_depth)].append([])                                    					  # for UNKNOWN
        listU, listL, listD = [], [], []
        for userid in self.tree[lr_bound_for_node[0]:(lr_bound_for_node[1]+1)]:
            if optimum_itemid not in self.rU[userid]:
                listU.append(userid)
            elif self.rU[userid][optimum_itemid] >= 4:
                listL.append(userid)
            elif self.rU[userid][optimum_itemid] <= 3:
                listD.append(userid)
        self.tree[lr_bound_for_node[0]:(lr_bound_for_node[1]+1)] = listL + listD + listU
        self.lr_bound[str(self.cur_depth)][-3] = [lr_bound_for_node[0], lr_bound_for_node[0]+len(listL)-1]                                                     # for LIKE
        self.lr_bound[str(self.cur_depth)][-2] = [lr_bound_for_node[0]+len(listL), lr_bound_for_node[0]+len(listL)+len(listD)-1]                                 # for DISLIKE
        self.lr_bound[str(self.cur_depth)][-1] = [lr_bound_for_node[0]+len(listL)+len(listD), lr_bound_for_node[0]+len(listL)+len(listD)+len(listU)-1]           # for UNKNOWN


        #### Generate Subtree of Node LIKE ####
        self.sum_cur_t = min_sumt[:, 0]
        self.sum_2_cur_t = min_sumt_2[:, 0]
        self.sum_cntt = min_cntt[:, 0]
        self.generate_decision_tree(self.lr_bound[str(self.cur_depth)][-3], chosen_id[:])    
        self.cur_depth -= 1

        #### Generate Subtree of Node DISLIKE ####
        self.sum_cur_t = min_sumt[:, 1]
        self.sum_2_cur_t = min_sumt_2[:, 1]
        self.sum_cntt = min_cntt[:, 1]
        self.generate_decision_tree(self.lr_bound[str(self.cur_depth)][-2], chosen_id[:])    
        self.cur_depth -= 1

        #### Generate Subtree of Node UNKNOWN ####
        self.sum_cur_t = min_sumt[:, 2]
        self.sum_2_cur_t = min_sumt_2[:, 2]
        self.sum_cntt = min_cntt[:, 2]
        self.generate_decision_tree(self.lr_bound[str(self.cur_depth)][-1], chosen_id[:])
        self.cur_depth -= 1

        #### Show Rating Progress ####
        for i in range(self.cur_depth - 1):
            print("┃", end="")        
        print("┏", end="")
        self.cur_node += 1
        print("Current depth: " + str(self.cur_depth) + "        %.2f%%" %(100*self.cur_node/self.node_num))


    def calculate_avg_rating_for_pesudo_user(self, pseudo_user_lst):
        '''ret_dict: dict {
            itemid0: rating0 
            itemid1: rating1
            ...				
        }'''
        cal_dict = {key: {'rating': 0, 'cnt': 0} for key in self.rI}
        ret_dict = {}
        for userid in pseudo_user_lst:
            for itemid, rating in self.rU[userid].items():
                cal_dict[itemid]['rating'] += rating
                cal_dict[itemid]['cnt'] += 1
        for itemid in cal_dict:
            if cal_dict[itemid]['cnt'] == 0:
                continue
            ret_dict[itemid] = cal_dict[itemid]['rating'] / cal_dict[itemid]['cnt']
        return ret_dict


    def generate_prediction_model(self):
        '''self.lr_bound: dict {
                    level 0: [[left_bound, right_bound]], users' bound for one level, each ele in dictionary represents one node
                    level 1: [[left_bound, right_bound], [left_bound, right_bound], [left_bound, right_bound]], 3
                    level 2: ..., 9
                } (bound means index)
        '''
        for level in self.lr_bound:
            self.user_profile.setdefault(level)
            train_lst = []
            for pseudo_user_bound, userid in zip(self.lr_bound[level], range(len(self.lr_bound[level]))):
                if pseudo_user_bound[0] > pseudo_user_bound[1]:
                    continue
                pseudo_user_lst = self.tree[pseudo_user_bound[0]:(pseudo_user_bound[1]+1)]
                pseudo_user_for_item = self.calculate_avg_rating_for_pesudo_user(pseudo_user_lst)
                train_lst += [(userid, int(key), float(value)) for key, value in pseudo_user_for_item.items()]
            self.user_profile[level], self.item_profile[level] = self.MF.matrix_factorization(train_lst)


    def build_model(self):
        #### Construct the tree & get the prediction model ####
        self.generate_decision_tree(self.lr_bound['0'][0], [])
        self.generate_prediction_model()


    def predict(self, new_user_ratings, pred_index):
        ''' new_user_ratings: list [
                       [itemid1, rating1],
                       [itemid2, rating2],
                       [itemid3, rating3],
                       [itemid4, rating4],
                       ... ] 
            pred_rating: array: (I,)
                            new user's rating for each item
         '''

        #### Find user profile for new user ####
        new_user_profile = np.array(self.user_profile[str(len(new_user_ratings))][pred_index])   # shape: (k,)
        new_item_profile = np.array(list(self.item_profile[str(len(new_user_ratings))].values()))               # shape: (I, k)

        #### Calculate predict rating ####
        pred_rating = {itemid: np.dot(new_item_profile[i], new_user_profile) \
            for itemid,i in zip(self.item_profile[str(len(new_user_ratings))], range(new_item_profile.shape[0]))}
        return pred_rating