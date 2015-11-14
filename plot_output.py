import matplotlib.pyplot as plt
import math
import numpy as np
import pylab
import os


mytype = "target"
path = "/Users/lvdandan/Desktop/myproject/mylasagne/result/"+mytype+"/"
output_path = "/Users/lvdandan/Desktop/myproject/mylasagne/result_img/"+mytype+"/"

for i in range(50):
	power = []
	file = open(path+str(i)+" "+mytype+".txt", 'r')
	for line in file.readlines():
		power.append(line.split()[0])       

	plt.plot(power)
	plt.grid()

	pylab.title("Power")
	pylab.xlabel("Time")
	pylab.ylabel("Power")
	plt.savefig(output_path+str(i)+" "+mytype+'.png', format='png')
	plt.clf()




  
  
