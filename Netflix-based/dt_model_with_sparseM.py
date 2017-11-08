import numpy as np
import pickle
from scipy.sparse import find

class DecisionTreeModel:
    def __init__(self, source, depth_threshold=10, plambda=7, MSP_item=200):
        '''
            sMatrix: I*U matrix
            depth_threshold: terminate depth
            plambda: regularization parameter
            self.rI: dict { itemid1, itemid2, itemid3 ... }
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
            every element represents ratings for one item, its order decide the users in tree nodes
        '''
        self.sMatrix = source
        self.real_item_num = source.shape[0]
        self.global_mean = 0  # global average of ratings   
        self.depth_threshold = depth_threshold
        self.plambda = plambda
        self.cur_depth = 0
        self.MSP_item = MSP_item

        #### Calculate rate of progress ####
        self.node_num = 0
        self.cur_node = 0
        for i in range(self.depth_threshold):
            self.node_num += 3 ** i

        #### Initiate Tree, lr_bound ####
        self.tree = list(range(1, self.sMatrix.shape[1]))
        self.split_item = []
        self.lr_bound = {'0': [[0, len(self.tree) - 1]]}

        #### Generate rI ####
        print("rI Generation start:")
        self.rI = list(set(source.nonzero()[0]))
        self.global_mean = source.sum()/source.getnnz()
        self.item_size = len(self.rI)
        self.user_size = len(self.tree)

        #### Generate bias, sum_cur_t, sum_2_cur_t ####
        self.biasU = {}
        self.sum_cur_t = np.zeros(self.real_item_num)
        self.sum_2_cur_t = np.zeros(self.real_item_num)
        self.sum_cntt = np.zeros(self.real_item_num)
        print("bias, sum_cur_t, sum_2_cur_t Generation start:")
        i = 0
        for userid in self.tree:
            if i % 5000 == 0:
                print("%.2f%%" % (100 * i / (0.75 * 480189)))
            i += 1

            self.biasU[userid] = (self.sMatrix.getcol(userid).sum() \
                                     + self.plambda * self.global_mean) /   \
                                 (self.plambda + self.sMatrix.getcol(userid).getnnz())
            user_all_rating_id = self.sMatrix.getcol(userid).nonzero()[0]
            # print('user_all_rating_id ', user_all_rating_id[:])
            user_all_rating = find(self.sMatrix.getcol(userid))[2]
            self.sum_cur_t[user_all_rating_id[:]] += user_all_rating[:] - self.biasU[userid]
            self.sum_2_cur_t[user_all_rating_id[:]] += (user_all_rating[:] - self.biasU[userid]) ** 2
            self.sum_cntt[user_all_rating_id[:]] += 1
        print("bias, sum_cur_t, sum_2_cur_t Generation DONE")

        print("Initiation DONE!")

    def calculate_error(self, sumt, sumt_2, cntt):
        ''' Calculate error for one item-split in one node '''
        Error_i = np.sum(sumt_2 - (sumt ** 2) / (cntt + 1e-9))

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
        for userid in self.tree[lr_bound_for_node[0]:(lr_bound_for_node[1] + 1)]:
            user_all_rating_id = self.sMatrix.getcol(userid).nonzero()[0]
            num_rec[user_all_rating_id[:]] += 1
        sub_item_id = np.argsort(-num_rec)[:self.MSP_item]

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
            user_rating_item_in_nodet = 
                list(set(self.tree[lr_bound_for_node[0]:(lr_bound_for_node[1] + 1)]).intersection(set(find(self.sMatrix[itemid, :])[1])))
            # user_rating_item_in_nodet = find(self.sMatrix[itemid, :])[self.tree[lr_bound_for_node[0]:(lr_bound_for_node[1] + 1)]]
            # user_rating_item_in_nodet = ([userid, self.rU[userid][itemid]] for userid in
            #                              self.tree[lr_bound_for_node[0]:(lr_bound_for_node[1] + 1)] if
            #                              itemid in self.rU[userid])
            sumt = np.zeros((self.real_item_num, 3))
            sumt_2 = np.zeros((self.real_item_num, 3))
            cntt = np.zeros((self.real_item_num, 3))
            for user in user_rating_item_in_nodet[1]:
                ''' user_all_rating: array [ [itemid11, rating11], [itemid12, rating12], ... ] '''
                user_all_rating_id = self.sMatrix.getcol(userid).nonzero()[0]
                user_all_rating = find(self.sMatrix.getcol(userid))[2]
                #### calculate sumtL for node LIKE ####
                if user_rating_item_in_nodet[2][user] >= 4:
                    sumt[user_all_rating_id[:], 0] += user_all_rating[:] - self.biasU[user]
                    sumt_2[user_all_rating_id[:], 0] += (user_all_rating[:] - self.biasU[user]) ** 2
                    cntt[user_all_rating_id[:], 0] += 1
                #### calculate sumtD for node DISLIKE ####
                elif user_rating_item_in_nodet[2][user] <= 3:
                    sumt[user_all_rating_id[:], 1] += user_all_rating[:] - self.biasU[user]
                    sumt_2[user_all_rating_id[:], 1] += (user_all_rating[:] - self.biasU[user]) ** 2
                    cntt[user_all_rating_id[:], 1] += 1
            #### calculate sumtU for node UNKNOWN ####
            sumt[:, 2] = self.sum_cur_t[:] - sumt[:, 0] - sumt[:, 1]
            sumt_2[:, 2] = self.sum_2_cur_t[:] - sumt_2[:, 0] - sumt_2[:, 1]
            cntt[:, 2] = self.sum_cntt[:] - cntt[:, 0] - cntt[:, 1]
            Error[itemid] = self.calculate_error(sumt, sumt_2, cntt)

            if min_Error == "None" or Error[itemid] < min_Error:
                min_sumt = sumt
                min_sumt_2 = sumt_2
                min_cntt = cntt
                min_Error = Error[itemid]
        #### Find optimum split-item ####
        optimum_itemid = min(Error, key=Error.get)
        if len(self.split_item) == self.cur_depth - 1:
            self.split_item.append([optimum_itemid])
        else:
            self.split_item[self.cur_depth - 1].append(optimum_itemid)
        # self.split_item.setdefault(str(self.cur_depth-1), []).append(optimum_itemid)
        chosen_id.append(optimum_itemid)

        #### sort tree ####
        self.lr_bound.setdefault(str(self.cur_depth), []).append([])  # for LIKE
        self.lr_bound[str(self.cur_depth)].append([])  # for DISLIKE
        self.lr_bound[str(self.cur_depth)].append([])  # for UNKNOWN
        listU, listL, listD = [], [], []
        for userid in self.tree[lr_bound_for_node[0]:(lr_bound_for_node[1] + 1)]:
            if optimum_itemid not in self.sMatrix.getcol(userid).nonzero()[0]:
                listU.append(userid)
            elif self.sMatrix[optimum_itemid, userid] >= 4:
                listL.append(userid)
            elif self.sMatrix[optimum_itemid, userid] <= 3:
                listD.append(userid)
        self.tree[lr_bound_for_node[0]:(lr_bound_for_node[1] + 1)] = listL + listD + listU
        self.lr_bound[str(self.cur_depth)][-3] = [lr_bound_for_node[0],
                                                  lr_bound_for_node[0] + len(listL) - 1]  # for LIKE
        self.lr_bound[str(self.cur_depth)][-2] = [lr_bound_for_node[0] + len(listL),
                                                  lr_bound_for_node[0] + len(listL) + len(listD) - 1]  # for DISLIKE
        self.lr_bound[str(self.cur_depth)][-1] = [lr_bound_for_node[0] + len(listL) + len(listD),
                                                  lr_bound_for_node[0] + len(listL) + len(listD) + len(
                                                      listU) - 1]  # for UNKNOWN

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
        print("Current depth: " + str(self.cur_depth) + "        %.2f%%" % (100 * self.cur_node / self.node_num))

    def build_model(self):
        #### Construct the tree & get the prediction model ####
        self.generate_decision_tree(self.lr_bound['0'][0], [])