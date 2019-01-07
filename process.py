import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import math
import numpy as np

numberImgs={}
#numberImgs["ship"]=[312,199,1223,1508,2725,5073,6181,7913,9018,9604]
#numberImgs["airplane"]=[1898, 2461, 3426, 5173, 180, 8563, 4408, 1573, 8105, 7684]

#numberImgs["dog"]=[239,1773,3000,7118,4917,9557,148,1548,640,1337]
#numberImgs["cat"]=[2804,760,1123,950,3479,4404,3799,4032,4718,3700]

#numberImgs["truck"]=[3344,970,349,2700,3841,7955]

numberImgs["automobile"]=[6, 9, 37, 66, 81, 82, 104, 105, 
114, 122, 131, 134, 161, 193, 201, 204, 231, 240, 241, 246, 
261, 283, 286, 290, 305, 308, 325, 330, 351, 363, 366, 369, 
390, 407, 414, 439, 440, 462, 490, 493, 494, 513, 540, 546, 
572, 594, 604, 619, 622, 629, 645, 656, 657, 659, 668, 677, 
723, 726, 736, 738, 753, 759, 764, 771, 781, 796, 801, 830, 
836, 844, 865, 869, 871, 887, 894, 895, 906, 915, 941, 947, 
953, 961, 962, 968, 973, 978, 987, 990, 997, 1005, 1006, 
1013, 1016, 1020, 1021, 1047, 1048, 1054, 1061, 1098, 1115, 
1131, 1134, 1141, 1144, 1156, 1158, 1174, 1176, 1182, 1185, 
1187, 1190, 1191, 1206, 1212, 1229, 1232, 1234, 1238, 1258, 
1260, 1270, 1288, 1301, 1306, 1313, 1332, 1335, 1340, 1354, 
1363, 1371, 1372, 1378, 1396, 1404, 1405, 1408, 1410, 1412, 
1413, 1414, 1435, 1437, 1444, 1457, 1458, 1464, 1467]

#numberImgs["deer"]=[7730,9205,4047,4112,4402,835,32,223,2805,1842]
#numberImgs["horse"]=[216,934,1615,2145,2062,4092,4871,9102,9334,9850]

attackTarget={}

#attackTarget["ship"]="airplane"
#attackTarget["airplane"]="ship"

#attackTarget["dog"]="cat"
#attackTarget["cat"]="dog"

#attackTarget["truck"]="automobile"
attackTarget["automobile"]="truck"

#attackTarget["deer"]="horse"
#attackTarget["horse"]="deer"

distance="L2"
bits=[2,8,16,32,64]
imagesSeries={}
pathFiles="/home/roki/GIT/QNNDeepGame/resultsCar2Truck16Agosto/"

def slope(x1, y1, x2, y2):
	m = (y2-y1)/(x2-x1)
	return m

plt.figure()
j=0
label="automobile"
results={}
imagesSeries={}
targetLabel=attackTarget[label]
delta=0.11
delta2=0.005

def checkSlope(slope1, slope2):
	if slope1*slope2>0:
		if abs(slope1-slope2)<delta:
			return True
	else:
		if abs(slope2)+abs(slope1)<delta2:
			return True
	return False
	
for image in numberImgs[label]: 
    imagesSeries[image]=[]
    for i in bits:
        #print i
        #print pathFiles+"cifar10"+str(image)+"Wbits"+str(i)+"Abits"+str(i)+".txt"
        with open(pathFiles+"cifar10"+str(image)+"Wbits"+str(i)+"Abits"+str(i)+".txt") as f:
            content = f.readlines()
        line0=content[0]
        words0=line0.replace("'","").split()
        predictionCorrect=words0.count(label)
        line_1=content[-1]
        words_1=line_1.replace("'","").split()
        #print "prediction count:"+str(predictionCorrect)
        if predictionCorrect==2:
            if targetLabel in words_1:
                #it means prediction is ok
                #and target attack is satisfied
                for line in content:
                    if distance in line:
                        val=float(line.split()[-1])
                        imagesSeries[image].append(val)
            else:
                imagesSeries[image].append(float('inf'))
        else: 
            imagesSeries[image].append(float('nan'))
listIS=list(imagesSeries.keys())
listISS=list(imagesSeries.keys())
for key in listIS:
    if key in imagesSeries.keys():
        valid=True
        for val in imagesSeries[key]:
            if math.isnan(val) or math.isinf(val):
                valid=False
        if valid:
            results[key]=[]
            for j in range(0,len(imagesSeries[key])):
                slope2=slope(0,imagesSeries[key][0],10,imagesSeries[key][1])
                slope8=slope(10,imagesSeries[key][1],20,imagesSeries[key][2])
                slope16=slope(20,imagesSeries[key][2],30,imagesSeries[key][3])
                slope32=slope(30,imagesSeries[key][3],40,imagesSeries[key][4])
                for similarImage in listISS:
									if similarImage in imagesSeries.keys():
											if similarImage!=key:
													valid=True
													for val in imagesSeries[similarImage]:
															if math.isnan(val) or math.isinf(val):
																	valid=False
													if valid:
															Simslope2=slope(0,imagesSeries[similarImage][0],10,imagesSeries[similarImage][1])
															Simslope8=slope(10,imagesSeries[similarImage][1],20,imagesSeries[similarImage][2])
															Simslope16=slope(20,imagesSeries[similarImage][2],30,imagesSeries[similarImage][3])
															Simslope32=slope(30,imagesSeries[similarImage][3],40,imagesSeries[similarImage][4])
															if checkSlope(slope2, Simslope2) and checkSlope(slope8, Simslope8) and checkSlope(slope16, Simslope16) and checkSlope(slope32, Simslope32):
																results[key].append(similarImage)
																del imagesSeries[similarImage]
print results
                            
                    
            
