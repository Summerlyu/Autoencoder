import matplotlib.pyplot as plt
import math
import numpy as np
import pylab
import os

path = "real_data/"

dishwasher = []
kettle = []
toaster = []
washmachine = []
aggre_len = 1200
washmachine 
def load_real_data():
	# file = open(path+str(1)+".dat", 'r')
	# for line in file.readlines():
	# 	dishwasher.append(int(line.split()[1]))       
	# dishwasher_train = dishwasher[600:1800]
	# dishwasher_valid = dishwasher[34250:35450]
	# dishwasher_test = dishwasher[68750:69950]

	# file = open(path+str(2)+".dat", 'r')
	# for line in file.readlines():
	# 	kettle.append(int(line.split()[1]))       
	# kettle_train = kettle[6000:6100]
	# kettle_valid = kettle[11000:11100]
	# kettle_test = kettle[19795:19895]

	# file = open(path+str(3)+".dat", 'r')
	# for line in file.readlines():
	# 	toaster.append(int(line.split()[1]))        
	# toaster_train = toaster[5040:5140]
	# toaster_valid = toaster[5490:5590]
	# toaster_test = toaster[7690:7790]



	# np.savetxt(path+"dishwasher_train.csv", dishwasher_train, delimiter=",")
	# np.savetxt(path+"kettle_train.csv", kettle_train, delimiter=",")
	# np.savetxt(path+"toaster_train.csv", toaster_train, delimiter=",")

	# np.savetxt(path+"dishwasher_valid.csv", dishwasher_valid, delimiter=",")
	# np.savetxt(path+"kettle_valid.csv", kettle_valid, delimiter=",")
	# np.savetxt(path+"toaster_valid.csv", toaster_valid, delimiter=",")

	# np.savetxt(path+"dishwasher_test.csv", dishwasher_test, delimiter=",")
	# np.savetxt(path+"kettle_test.csv", kettle_test, delimiter=",")
	# np.savetxt(path+"toaster_test.csv", toaster_test, delimiter=",")

	file = open(path+str(4)+".dat", 'r')
	for line in file.readlines():
		washmachine.append(int(line.split()[1]))        
	washmachine_train = washmachine[6840:8040]
	washmachine_valid = washmachine[26330:27530]
	washmachine_test = washmachine[51470:52670]

	np.savetxt(path+"washmachine_train.csv", washmachine_train, delimiter=",")
	np.savetxt(path+"washmachine_valid.csv", washmachine_valid, delimiter=",")
	np.savetxt(path+"washmachine_test.csv", washmachine_test, delimiter=",")


if __name__ == '__main__':
	ret_lrd = load_real_data()

