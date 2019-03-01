import shlex
import subprocess
import sys
import multiprocessing

'''
For every image, an adversarial sample is crated for every quantization level (2, 4, .. 64).
Creation of an adversarial sample is isolated in a separate process.
It follows that for every image, there will be 6 different processes.
But for creation of an adversarial sample for the next image, it is not necessary to wait for the previous image to finish.
So the creation of adversarial samples for the next image starts in parallel with the creation of adv samples for the previous image and so on.
'''


def process_image(cmd):
	cmd_original = cmd
	print("Starting: " + str(cmd_original))
	cmd = shlex.split(cmd)
	r = subprocess.Popen(cmd, shell=False)
	r.wait()
	print("Finished: " + str(cmd_original))



def main():
	pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
	print("pool size: ")
	quantization = [2, 4, 8, 16, 32, 64]

	print(sys.argv[1])
	seed = int(sys.argv[1])
	startPoint = int(sys.argv[2])
	lenTest = int(sys.argv[3])
	dataset = str(sys.argv[4])

	for sourceImage in range(startPoint, lenTest):
		print(sourceImage)
		index0 = sourceImage
		for q in quantization:
			exe = "python main.py " + dataset + " ub cooperative "+str(index0)+" L2 10 1 "+str(q)+" "+str(q)+" "+str(seed)
			pool.apply_async(process_image, [exe])

	pool.close()
	pool.join()


if __name__=="__main__":
	main()