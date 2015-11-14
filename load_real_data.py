import matplotlib.pyplot as plt
import math
import numpy as np
import pylab
import os
import pandas
import theano
import lasagne
from sklearn import preprocessing
import theano.tensor as T

file_path = "real_data/"
dishwasher = []		# 1200
kettle = []			# 100
washmachine = [] 	# 1200
toaster = []		# 100
aggre_len = 1200
num_sample = 10000

def _load_real_data(filepath):
	data_list = []
	data_df = pandas.read_csv(filepath, index_col=False, header=None)
	for row in enumerate(data_df.values):
		data_list.append(float(row[1]))
	data_list = preprocessing.scale(data_list)
	# print("data_list:{}".format(len(data_list)))
	# print("_______")
	return data_list

def load_real_data():
	dishwasher_train = _load_real_data(file_path+'dishwasher_train.csv')
	dishwasher_valid = _load_real_data(file_path+'dishwasher_valid.csv')
	dishwasher_test = _load_real_data(file_path+'dishwasher_test.csv')

	kettle_train = _load_real_data(file_path+'kettle_train.csv')
	kettle_valid = _load_real_data(file_path+'kettle_valid.csv')
	kettle_test = _load_real_data(file_path+'kettle_test.csv')

	washmachine_train = _load_real_data(file_path+'washmachine_train.csv')
	washmachine_valid = _load_real_data(file_path+'washmachine_valid.csv')
	washmachine_test = _load_real_data(file_path+'washmachine_test.csv')

	toaster_train = _load_real_data(file_path+'toaster_train.csv')
	toaster_valid = _load_real_data(file_path+'toaster_valid.csv')
	toaster_test = _load_real_data(file_path+'toaster_test.csv')

	ret=dict(
	        dw_train = list(dishwasher_train),
	        dw_valid = list(dishwasher_valid),
	        dw_test = list(dishwasher_test),

	        ket_train = list(kettle_train),
	        ket_valid = list(kettle_valid),
	        ket_test = list(kettle_test),

	        wm_train = list(washmachine_train),
	        wm_valid = list(washmachine_valid),
	        wm_test = list(washmachine_test),

	        toa_train = list(toaster_train),
	        toa_valid = list(toaster_valid),
	        toa_test = list(toaster_test),	     
	)
	return ret

def aggre_lists_value(list_parent, list_son, start):
	j=0
	for i in range(start, start+len(list_son)):
		list_parent[i] += list_son[j]
		j=j+1
	return list_parent

def generate_inout(list_a, list_b, list_c, list_d):
	# r_ind_a = np.random.randint(0, aggre_len - len(list_a) - 1)
	r_ind_b = np.random.randint(0, aggre_len - len(list_b) - 1)
	# r_ind_c = np.random.randint(0, aggre_len - len(list_c) - 1)
	r_ind_d = np.random.randint(0, aggre_len - len(list_d) - 1)

	zero_list = [0]*aggre_len
	# flag = np.random.rand()
	# print("flag:{}".format(flag))
	# if flag < 0.95:
	# 	list_c = [0]*aggre_len

	aggr_list = aggre_lists_value(zero_list, list_a, 0)
	aggr_list = aggre_lists_value(aggr_list, list_b, r_ind_b)
	aggr_list = aggre_lists_value(aggr_list, list_c, 0)
	aggr_list = aggre_lists_value(aggr_list, list_d, r_ind_d)

	# zero_list = [0]*aggre_len
	# new_list_a = aggre_lists_value(zero_list, list_a, r_ind_a)

	zero_list = [0]*aggre_len
	new_list_b = aggre_lists_value(zero_list, list_b, r_ind_b)

	# zero_list = [0]*aggre_len
	# new_list_c = aggre_lists_value(zero_list, list_c, r_ind_c)

	zero_list = [0]*aggre_len
	new_list_d = aggre_lists_value(zero_list, list_d, r_ind_d)

	ret_gi = dict(
		aggr_list = aggr_list,
		new_list_a = list_a,
		new_list_b = new_list_b, 
		new_list_c = list_c,
		new_list_d = new_list_d,
		)
	return ret_gi

def sample_reproduction(dw, ket, wm, toa):
	aggr_lists = []
	new_lists_a = []
	new_lists_b = []
	new_lists_c = []
	new_lists_d = []

	for i in range(num_sample):
		ret_gi = generate_inout(dw, ket, wm, toa)
		aggr_lists.append(ret_gi["aggr_list"])
		new_lists_a.append(ret_gi["new_list_a"])
		new_lists_b.append(ret_gi["new_list_b"])
		new_lists_c.append(ret_gi["new_list_c"])
		new_lists_d.append(ret_gi["new_list_d"])

	ret_sr = dict(
		X = aggr_lists,
		y_a = new_lists_a,
		y_b = new_lists_b,
		y_c = new_lists_c,
		y_d = new_lists_d,
		)
	return ret_sr

def load_real_data_main():
	# os.system("clear")
	ret_lrd = load_real_data()

	ret_train = sample_reproduction(ret_lrd["dw_train"], ret_lrd["ket_train"], ret_lrd["wm_train"], ret_lrd["toa_train"])
	ret_valid = sample_reproduction(ret_lrd["dw_valid"], ret_lrd["ket_valid"], ret_lrd["wm_valid"], ret_lrd["toa_valid"])
	ret_test = sample_reproduction(ret_lrd["dw_test"], ret_lrd["ket_test"], ret_lrd["wm_test"], ret_lrd["toa_test"])


	X_train = np.array(ret_train["X"])
	X_valid = np.array(ret_valid["X"])
	X_test = np.array(ret_test["X"])


	# Dishwasher
	# y_train=np.array(ret_train["y_a"])
	# y_valid=np.array(ret_valid["y_a"])
	# y_test=np.array(ret_test["y_a"])

	# # Kettle
	# y_train=np.array(ret_train["y_b"])
	# y_valid=np.array(ret_valid["y_b"])
	# y_test=np.array(ret_test["y_b"])

	# Wash machine
	y_train=np.array(ret_train["y_c"])
	y_valid=np.array(ret_valid["y_c"])
	y_test=np.array(ret_test["y_c"])

	# # Toaster
	# y_train=np.array(ret_train["y_d"])
	# y_valid=np.array(ret_valid["y_d"])
	# y_test=np.array(ret_test["y_d"])

	# plt.plot(y_test[0])
	# plt.grid()
	# pylab.xlabel("Time")
	# pylab.ylabel("Kettle y_train")
	# plt.show()
	
	ret_lrfm = dict(
			# X_valid_l = X_valid.tolist(),
	        X_train=theano.shared(lasagne.utils.floatX(X_train)),
	        X_valid=theano.shared(lasagne.utils.floatX(X_valid)),
	        X_test=theano.shared(lasagne.utils.floatX(X_test)),

			y_train=theano.shared(lasagne.utils.floatX(y_train)),
			y_valid=theano.shared(lasagne.utils.floatX(y_valid)),
			y_test=theano.shared(lasagne.utils.floatX(y_test)),

	        num_train_data=X_train.shape[0],
	        num_valid_data=X_valid.shape[0],
	        num_test_data=X_test.shape[0],

	        input_d=X_train.shape[1],
	        output_d=aggre_len
		)
	return ret_lrfm


if __name__ == '__main__':
	# ret_lrd = load_real_data()
	# ret_gi = generate_inout(ret_lrd["dw_test"], ret_lrd["ket_test"], ret_lrd["toa_test"])
	# plt.subplot(4, 1, 1)
	# plt.plot(ret_gi["aggr_list"])
	# plt.grid()
	# pylab.xlabel("Time")
	# pylab.ylabel("aggr_list")


	# plt.subplot(4, 1, 2)
	# plt.plot(ret_gi["new_list_a"])
	# plt.grid()
	# pylab.xlabel("Time")
	# pylab.ylabel("Wash Machine")


	# plt.subplot(4, 1, 3)
	# plt.plot(ret_gi["new_list_b"])
	# plt.grid()
	# pylab.xlabel("Time")
	# pylab.ylabel("Kettle")


	# plt.subplot(4, 1, 4)
	# plt.plot(ret_gi["new_list_c"])
	# plt.grid()
	# pylab.xlabel("Time")
	# pylab.ylabel("Toaster")
	# plt.show()
	# plt.close()

	load_real_data_main()

	# print("----------load data----------")
	# ret = load_real_data_main()
	# X_valid_l = ret["X_valid_l"]
	# print(X_valid_l)
	# print(len(X_valid_l))