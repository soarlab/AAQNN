import shlex
import subprocess
import sys
from random import seed as seedR
from numpy.random import seed
from tensorflow import set_random_seed


seedNumber=int(sys.argv[1]) #10
set_random_seed(seedNumber)
seed(seedNumber)
seedR(seedNumber)

init=int(sys.argv[2]) #0
end=int(sys.argv[3]) #10

dataset=str(sys.argv[4])
#dataset="cifar10"
#dataset="mnist"
#dataset="fashion"

processes=[]
exe="python executor.py "+str(seedNumber)+" "+str(init)+" "+str(end)+ " "+ dataset
print(exe)
exe=shlex.split(exe)
p=subprocess.Popen(exe,shell=False)
processes.append(p)
p.wait()
print("Done with images from "+str(init)+" to "+str(end))
