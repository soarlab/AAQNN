import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'font.size': 15})
#seed_10_cifar10_2066_Wbits16Abits16.txt

seed=10
distance="L2"
bits=[2,8,16,32,64]
pathFiles="results/"
dictBits={}
for val in bits:
	dictBits[val]=[]

for image in range(0,10000):
	for i in bits:
		try:
			with open(pathFiles+"seed_"+str(seed)+"_cifar10_"+str(image)+"_Wbits"+str(i)+"Abits"+str(i)+".txt") as f:
				content = f.readlines()
			line0=content[0]
			
			splitPrediction=line0.split("'")
			prediction=splitPrediction[1]
			print(prediction)
			splitSpace=line0.replace("'","").split()
			print(splitSpace)
			predictionCorrect=splitSpace.count(prediction)
			
			if predictionCorrect==2:
				tmp=False
				for line in content:
					if distance in line:
						val=float(line.split()[-1])
						dictBits[i].append(val)
						tmp=True
				if not tmp:
					#no attack for L2<10, try with 10
					dictBits[i].append(10.0)
			else:
				#misclassification without attack, try with 0
				dictBits[i].append(0.0)
		except IOError:
			pass
			
data_to_plot = [dictBits[2], dictBits[8], dictBits[16], dictBits[32], dictBits[64]]
fig = plt.figure(1, figsize=(9, 6))

# Create an axes instance
ax = fig.add_subplot(111)

# Create the boxplot
bp = ax.boxplot(data_to_plot,0,'gD')

ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.set_xticklabels(['2', '8', '16', '32', '64'])
ax.set_ylim([-3,12])
plt.show()
