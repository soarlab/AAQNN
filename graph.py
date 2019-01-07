import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import math
import numpy as np

matplotlib.rcParams.update({'font.size': 15})


numberImgs={}
#numberImgs["ship"]=[312,199,1223,1508,2725,5073,6181,7913,9018,9604]
#numberImgs["airplane"]=[1898, 2461, 3426, 5173, 180, 8563, 4408, 1573, 8105, 7684]

#numberImgs["dog"]=[239,1773,3000,7118,4917,9557,148,1548,640,1337]
#numberImgs["cat"]=[2804,760,1123,950,3479,4404,3799,4032,4718,3700]

#numberImgs["truck"]=[3344,970,349,2700,3841,7955]

numberImgs["automobile"]=[1176, 1301, 987]
'''
[6, 9, 37, 66, 81, 82, 104, 105, 
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
'''
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
indexs = [x * 10 for x in range(0,len(bits))]
print indexs
imagesSeries={}
pathFiles="/home/roki/GIT/QNNDeepGame/resultsCar2Truck16Agosto/"

plt.figure()
j=0
for label in numberImgs.keys():
    #print label
    imagesSeries={}
    targetLabel=attackTarget[label]
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
        #print imagesSeries
        #plt.figure()
        #plt.xticks(range(1,len(bits)+1),bits)
        color = cm.rainbow(np.linspace(0, 1, len(numberImgs["automobile"])))
        #for key in imagesSeries:
        key=image

        plot=True
        for val in imagesSeries[key]:
            if math.isnan(val) or math.isinf(val):
                plot=False
        if plot:
            print "j:"+str(j)
            print image
            print imagesSeries[key]
            axes = plt.gca()
            #axes.set_ylim([0,10])
            tmp,=plt.plot(indexs, imagesSeries[key], '-o', color=color[j],label=key)
            plt.title("From: "+label+" to:"+targetLabel)
            plt.legend(loc=0, prop={'size': 15})            
            plt.xticks(indexs, bits)
            #plt.annotate(str(image),xy=(1,imagesSeries[key][0]))
            plt.annotate(str(image),xy=(64,imagesSeries[key][4]))
            plt.xlabel('QUANTIZATION')
            plt.ylabel('L 2')
            j=j+1
            plt.show(block=False)
            val=raw_input()
            if val=="r":
                tmp.remove()
print "DONE"
plt.show()



