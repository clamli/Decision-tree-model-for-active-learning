############# Input Package ################
from scipy.sparse import load_npz
import dt_model as dt
import tool_function as tf
############################################

############### Load Data ##################
rating_matrix_csc = load_npz('./netflix/sparse_matrix.npz').tocsc()
print("file load DONE")
############################################

############### Build Tree #################
''' Save to file 'tree.pkl' '''
start = 0
end = int(rating_matrix_csc.shape[1] * 0.75)
dtmodel_realdata = dt.DecisionTreeModel(rating_matrix_csc[:, start:end], depth_threshold = 10)
dtmodel_realdata.build_model()
o_tree = open('./tree_data_structure/tree.pkl', 'wb')
pickle.dump(dtmodel_realdata, o_tree)
o_tree.close()
############################################

######################## Build Predict Model #########################
plambda_candidates = {}
inputf = open('./tree_data_structure/tree.pkl', 'rb')
dtmodel_realdata = pickle.load(inputf)
for level in dtmodel_realdata.lr_bound:
	plambda_candidates[level] = list(np.arange(0.001, 0.05, 0.0005))
prediction_model = tf.generate_prediction_parameter(dtmodel_realdata, plambda_candidates)
######################################################################

######################### Test for New-user ##########################
rmse_result = tf.pred_RMSE_for_new_item(dtmodel_realdata, prediction_model, rating_matrix_csc[:, end:])
######################################################################



    