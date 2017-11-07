############# Input Package ################
from scipy.sparse import load_npz
import dt_model as dt
import tool_function as tf
import klepto
############################################

############### Load Data ##################
rating_matrix_csc = load_npz('./netflix/sparse_matrix_100%.npz').tocsc()
rating_matrix_val_csc = load_npz('./netflix/sparse_matrix_validation_75%.npz').tocsc()
print("file load DONE")
############################################

############### Build Tree #################
start = 0
end = int(rating_matrix_csc.shape[1] * 0.75)
dtmodel_realdata = dt.DecisionTreeModel(rating_matrix_csc[:, start:end], depth_threshold = 10)
dtmodel_realdata.build_model()
Tree = klepto.archives.dir_archive('treeFile', cached=True, serialized=True)
Tree['lr_bound'] = dtmodel_realdata.lr_bound
Tree['tree'] = dtmodel_realdata.tree
Tree['split_item'] = dtmodel_realdata.split_item
Tree['rI'] = dtmodel_realdata.rI
Tree.dump()
Tree.clear()
############################################

######################## Build Predict Model #########################
Tree = klepto.archives.dir_archive('treeFile', cached=True, serialized=True)
Tree.load('treeFile')
plambda_candidates = {}
for level in Tree['lr_bound']:
	plambda_candidates[level] = list(np.arange(0.001, 0.05, 0.0005))
prediction_model = tf.generate_prediction_model(Tree['lr_bound'], Tree['tree'], Tree['rI'], rating_matrix_csc[:, start:end].tocsr(), plambda_candidates, rating_matrix_val_csc)
######################################################################

######################### Test for New-user ##########################
rmse_result = tf.pred_RMSE_for_new_user(Tree['split_item'], rI, prediction_model, rating_matrix_csc[:, end:])
######################################################################



    