def RMSE(real_rating, pred_rating, rated_item):
	rmse, cnt = 0, 0
	for itemid, rating in real_rating.items():
		if itemid not in rated_item:
			rmse += (pred_rating[itemid] - rating)**2
			cnt += 1
	return (rmse/cnt)**0.5

def pred_RMSE_for_new_item(fdtmodel, sM_testing):
	'''
		fdtmodel: FDT class instance
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
	x = find(sM_testing)
	itemset = x[0]
	userset = x[1]
	User = {}
	for itemid, userid in zip(itemset, userset):
		if itemid in fdtmodel.rI:
			User.setdefault(userid, {})[itemid] = sM_testing[itemid, userid]

	rmse = 0
	for userid in User:
		pred_index = 0
		new_user_ratings = []
		rated_item = []
		for level in range(len(fdtmodel.split_item)):
			if fdtmodel.split_item[level][pred_index] not in User[userid]:
				tmp_pred_index = 3*pred_index + 2
				if tmp_pred_index in fdtmodel.user_profile[str(int(level)+1)]:
					new_user_ratings.append([fdtmodel.split_item[level][pred_index], 0])
					pred_index = tmp_pred_index
				else:
					break
			elif User[userid][fdtmodel.split_item[level][pred_index]] >= 4:
				tmp_pred_index = 3*pred_index
				if tmp_pred_index in fdtmodel.user_profile[str(int(level)+1)]:
					rated_item.append(fdtmodel.split_item[level][pred_index])
					new_user_ratings.append([fdtmodel.split_item[level][pred_index], User[userid][fdtmodel.split_item[level][pred_index]]])
					pred_index = tmp_pred_index
				else:
					break
			elif User[userid][fdtmodel.split_item[level][pred_index]] <= 3:
				tmp_pred_index = 3*pred_index + 1
				if tmp_pred_index in fdtmodel.user_profile[str(int(level)+1)]:
					rated_item.append(fdtmodel.split_item[level][pred_index])
					new_user_ratings.append([fdtmodel.split_item[level][pred_index], User[userid][fdtmodel.split_item[level][pred_index]]])
					pred_index = tmp_pred_index
				else:
					break
		pred_rating = fdtmodel.predict(new_user_ratings, pred_index)
		rmse += RMSE(User[userid], pred_rating, rated_item)

	return rmse / len(User)