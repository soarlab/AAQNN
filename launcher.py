import shlex
import subprocess
import os
import sys
from random import seed as seedR
from numpy.random import seed
from tensorflow import set_random_seed

seedNumber=10
set_random_seed(seedNumber)
seed(seedNumber)
seedR(seedNumber)

init=0
end=1

#dataset="cifar10"
#dataset="mnist"
dataset="fashion"


processes=[]
for i in range(0,1): 
	exe="python executor.py "+str(seedNumber)+" "+str(init)+" "+str(end)+ " "+ dataset
	print exe
	exe=shlex.split(exe)
	p=subprocess.Popen(exe,shell=False)
	processes.append(p)
for p in processes:
		p.wait()
