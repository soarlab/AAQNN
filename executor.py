import shlex
import subprocess
import os
import sys


quantization=[2,4,8,16,32,64]

print sys.argv[1]
seed=int(sys.argv[1])
startPoint=int(sys.argv[2])
lenTest=int(sys.argv[3])
dataset=str(sys.argv[4])

for sourceImage in range(startPoint,lenTest):
	print sourceImage
	index0=sourceImage
	processes=[]
	for q in quantization:
		exe="python main.py "+dataset+" ub cooperative "+str(index0)+" L2 10 1 "+str(q)+" "+str(q)+" "+str(seed)
		print exe
		exe=shlex.split(exe)
		p=subprocess.Popen(exe,shell=False)
		processes.append(p)
	for p in processes:
		p.wait()
	print ("Done with: "+str(index0))
	
