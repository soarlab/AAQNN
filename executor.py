import shlex
import subprocess
import os
import sys

'''
For every image, an adversarial sample is crated for every quantization level (2, 4, .. 64).
Creation of an adversarial sample is isolated in a separate process.
It follows that for every image, there will be 6 different processes.
But for creation of an adversarial sample for the next image, it is not necessary to wait for the previous image to finish.
So the creation of adversarial samples for the next image starts in parallel with the creation of adv samples for the previous image and so on.
'''

quantization=[2,4,8,16,32,64]

# controls total number of concurrent process that is run, if set too high, not-enough-memory exception could occur
MAX_NUMBER_OF_CONCURRENT_PROCESSES = 60

print(sys.argv[1])
seed=int(sys.argv[1])
startPoint=int(sys.argv[2])
lenTest=int(sys.argv[3])
dataset=str(sys.argv[4])

processes={}

for sourceImage in range(startPoint,lenTest):
	print(sourceImage)
	index0=sourceImage
	for q in quantization:
		exe="python main.py "+dataset+" ub cooperative "+str(index0)+" L2 10 1 "+str(q)+" "+str(q)+" "+str(seed)
		print(exe)
		exe=shlex.split(exe)
		p=subprocess.Popen(exe,shell=False)
		processes[p.pid]=1	#.append(p)
		if len(processes.keys())>=MAX_NUMBER_OF_CONCURRENT_PROCESSES:
			(pid,exitstat) = os.wait()
			del processes[pid]
