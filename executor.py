import shlex
import subprocess
import os
import sys


quantization=[2,4,8,16,32,64]

concurrentProcesses=len(quantization)

print sys.argv[1]
seed=int(sys.argv[1])
startPoint=int(sys.argv[2])
lenTest=int(sys.argv[3])
dataset=str(sys.argv[4])

processes={}

for sourceImage in range(startPoint,lenTest):
	print sourceImage
	index0=sourceImage
	for q in quantization:
		exe="python main.py "+dataset+" ub cooperative "+str(index0)+" L2 10 1 "+str(q)+" "+str(q)+" "+str(seed)
		print exe
		exe=shlex.split(exe)
		p=subprocess.Popen(exe,shell=False)
		processes[p.pid]=1	#.append(p)
		if len(processes.keys())>=concurrentProcesses:
			(pid,exitstat) = os.wait()
			del processes[pid]
