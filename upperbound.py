from __future__ import print_function
from NeuralNetwork import *
from DataSet import *
from CompetitiveMCTS import *
from CooperativeMCTS import *


def upperbound(dataSetName, bound, tau, gameType, image_index, eta, wbits, abits,nameFile,seed):
	start_time = time.time()
	
	outF=open("results/"+nameFile,"w+")
	
	MCTS_all_maximal_time = 500
	MCTS_level_maximal_time = 60
		
	NN = NeuralNetwork(dataSetName, abits, wbits, 'full-qnn',seed)
	NN.train_network_QNN()
	print("Dataset is %s." % NN.data_set)
	dataset = DataSet(dataSetName, 'testing')    

	realLabel=dataset.get_True_Label(image_index)
	image = dataset.get_input(image_index)
	(label, confident) = NN.predict(image)
	predicted = NN.get_label(int(label))
	print("Prediction on input with index %s, whose class predicted is '%s' (true value %s ) and the confidence is %s."
		  % (image_index, predicted, NN.get_label(int(realLabel)), confident))
	outF.write("Prediction on input with index %s, whose class predicted is '%s' (true value %s ) and the confidence is %s.\n"
		  % (image_index, predicted, NN.get_label(int(realLabel)), confident))
	print("the second player is %s." % gameType)
	
	outF.write("the second player is %s.\n" % gameType)
	
	if not predicted== NN.get_label(int(realLabel)):
		print ("NN misclassification without attack")
		outF.write("NN misclassification without attack")
		exit(0)
	# tau = 1
	outF.flush()
	# choose between "cooperative" and "competitive"
	if gameType == 'cooperative':
		mctsInstance = MCTSCooperative(dataSetName, NN, image_index, image, tau, eta)#, target_class)
		mctsInstance.initialiseMoves()
	
		start_time_all = time.time()
		runningTime_all = 0
		start_time_level = time.time()
		runningTime_level = 0 
		currentBest = eta[1]
		while runningTime_all <= MCTS_all_maximal_time:
			print ("Time: "+str(runningTime_all))
			'''
			if runningTime_level > MCTS_level_maximal_time: 
				bestChild = mctsInstance.bestChild(mctsInstance.rootIndex)
				# pick the current best move to take  
				mctsInstance.makeOneMove(bestChild)
				start_time_level = time.time()
			'''
			 
			# Here are three steps for MCTS
			(leafNode, availableActions) = mctsInstance.treeTraversal(mctsInstance.rootIndex)
			newNodes = mctsInstance.initialiseExplorationNode(leafNode, availableActions)
			for node in newNodes:
				(_, value) = mctsInstance.sampling(node, availableActions)
				mctsInstance.backPropagation(node, value)
			if currentBest > mctsInstance.bestCase[0]:
				print("best distance up to now is %s" % (str(mctsInstance.bestCase[0])))
				currentBest = mctsInstance.bestCase[0]
			bestChild = mctsInstance.bestChild(mctsInstance.rootIndex)
	
			# store the current best
			(_, bestManipulation) = mctsInstance.bestCase
			image1 = mctsInstance.applyManipulation(bestManipulation)
			path0 = "%s_pic/%s_currentBest.png" % (dataSetName, image_index)
			NN.save_input(image1, path0)
	
			runningTime_all = time.time() - start_time_all
			runningTime_level = time.time() - start_time_level
	
		(_, bestManipulation) = mctsInstance.bestCase
	
		print("the number of sampling: %s" % mctsInstance.numOfSampling)
		print("the number of adversarial examples: %s\n" % mctsInstance.numAdv)
	
		image1 = mctsInstance.applyManipulation(bestManipulation)
		(newClass, newConfident) = NN.predict(image1)
		newClassStr = NN.get_label(int(newClass))
		
		if newClass != label:# and str(target_class)==str(NN.get_label(int(newClass))):
			path0 = "%s_pic/%s_%s_modified_into_%s_with_confidence_%s.png" % (
				dataSetName, image_index, NN.get_label(int(realLabel)), newClassStr, newConfident)
			NN.save_input(image1, path0)
			path0 = "%s_pic/%s_diff.png" % (dataSetName, image_index)
			NN.save_input(np.absolute(image - image1), path0)
			print("\nfound an adversary image within pre-specified bounded computational resource. "
				  "The following is its information: ")
			outF.write("\nfound an adversary image within pre-specified bounded computational resource. "
				  "The following is its information: \n")
				  
			print("difference between images: %s" % (diffImage(image, image1)))
		
			outF.write("difference between images: %s\n" % (diffImage(image, image1)))
	
			print("number of adversarial examples found: %s\n" % mctsInstance.numAdv)
			
			outF.write("number of adversarial examples found: %s\n" % mctsInstance.numAdv)
	
			l2dist = l2Distance(mctsInstance.image, image1)
			l1dist = l1Distance(mctsInstance.image, image1)
			l0dist = l0Distance(mctsInstance.image, image1)
			percent = diffPercent(mctsInstance.image, image1)
			print("L2 distance %s" % l2dist)
			outF.write("L2 distance %s\n" % l2dist)
			
			print("L1 distance %s" % l1dist)
			outF.write("L1 distance %s\n" % l1dist)
			
			print("L0 distance %s" % l0dist)
			outF.write("L0 distance %s\n" % l0dist)
			
			print("manipulated percentage distance %s" % percent)
			outF.write("manipulated percentage distance %s\n" % percent)
			
			print("class is changed into '%s' with confidence %s\n" % (newClassStr, newConfident))
			outF.write("class is changed into '%s' with confidence %s\n" % (newClassStr, newConfident))
			
			return time.time() - start_time_all, newConfident, percent, l2dist, l1dist, l0dist, 0
	
		else:
			print("\nfailed to find an adversary image within pre-specified bounded computational resource. ")
			outF.write("\nfailed to find an adversary image within pre-specified bounded computational resource. \n")
			return 0, 0, 0, 0, 0, 0, 0
		outF.flush()
		outF.close()
	elif gameType == 'competitive':
	
		mctsInstance = MCTSCompetitive(dataSetName, NN, image_index, image, tau, eta)
		mctsInstance.initialiseMoves()
	
		start_time_all = time.time()
		runningTime_all = 0
		currentBest = eta[1]
		currentBestIndex = 0
		while runningTime_all <= MCTS_all_maximal_time:
	
			(leafNode, availableActions) = mctsInstance.treeTraversal(mctsInstance.rootIndex)
			newNodes = mctsInstance.initialiseExplorationNode(leafNode, availableActions)
			for node in newNodes:
				(_, value) = mctsInstance.sampling(node, availableActions)
				mctsInstance.backPropagation(node, value)
			if currentBest > mctsInstance.bestCase[0]:
				print("best distance up to now is %s" % (str(mctsInstance.bestCase[0])))
				currentBest = mctsInstance.bestCase[0]
				currentBestIndex += 1
	
			# store the current best
			(_, bestManipulation) = mctsInstance.bestCase
			image1 = mctsInstance.applyManipulation(bestManipulation)
			path0 = "%s_pic/%s_currentBest_%s.png" % (dataSetName, image_index, currentBestIndex)
			NN.save_input(image1, path0)
	
			runningTime_all = time.time() - start_time_all
	
		(bestValue, bestManipulation) = mctsInstance.bestCase
	
		print("the number of sampling: %s" % mctsInstance.numOfSampling)
		print("the number of adversarial examples: %s\n" % mctsInstance.numAdv)
	
		print("the number of max features is %s" % mctsInstance.bestFeatures()[0])
		maxfeatures = mctsInstance.bestFeatures()[0]
	
		if bestValue < eta[1]:
	
			image1 = mctsInstance.applyManipulation(bestManipulation)
			(newClass, newConfident) = NN.predict(image1)
			newClassStr = NN.get_label(int(newClass))
	
			if newClass != label:
				path0 = "%s_pic/%s_%s_modified_into_%s_with_confidence_%s.png" % (
					dataSetName, image_index, NN.get_label(int(realLabel)), newClassStr, newConfident)
				NN.save_input(image1, path0)
				path0 = "%s_pic/%s_diff.png" % (dataSetName, image_index)
				NN.save_input(np.absolute(image - image1), path0)
				print("\nfound an adversary image within pre-specified bounded computational resource. "
					  "The following is its information: ")
				print("difference between images: %s" % (diffImage(image, image1)))
	
				print("number of adversarial examples found: %s" % mctsInstance.numAdv)
	
				l2dist = l2Distance(mctsInstance.image, image1)
				l1dist = l1Distance(mctsInstance.image, image1)
				l0dist = l0Distance(mctsInstance.image, image1)
				percent = diffPercent(mctsInstance.image, image1)
				print("L2 distance %s" % l2dist)
				print("L1 distance %s" % l1dist)
				print("L0 distance %s" % l0dist)
				print("manipulated percentage distance %s" % percent)
				print("class is changed into '%s' with confidence %s\n" % (newClassStr, newConfident))
	
				return time.time() - start_time_all, newConfident, percent, l2dist, l1dist, l0dist, maxfeatures
	
			else:
				print("\nthe robustness of the (input, model) is under control, "
					  "with the first player is able to defeat the second player "
					  "who aims to find adversarial example by "
					  "playing suitable strategies on selecting features. ")
				return 0, 0, 0, 0, 0, 0, 0
	
		else:
	
			print("\nthe robustness of the (input, model) is under control, "
				  "with the first player is able to defeat the second player "
				  "who aims to find adversarial example by "
				  "playing suitable strategies on selecting features. ")
			return 0, 0, 0, 0, 0, 0, 0
	
	else:
		print("Unrecognised game type. Try 'cooperative' or 'competitive'.")
	
	runningTime = time.time() - start_time


'''

    if gameType == 'cooperative':
        mctsInstance = MCTSCooperative(dataSetName, NN, image_index, image, tau, eta)
        mctsInstance.initialiseMoves()

        start_time_all = time.time()
        runningTime_all = 0
        numberOfMoves = 0
        while (not mctsInstance.terminalNode(mctsInstance.rootIndex) and
               not mctsInstance.terminatedByEta(mctsInstance.rootIndex) and
               runningTime_all <= MCTS_all_maximal_time):
            print("the number of moves we have made up to now: %s" % numberOfMoves)
            l2dist = mctsInstance.l2Dist(mctsInstance.rootIndex)
            l1dist = mctsInstance.l1Dist(mctsInstance.rootIndex)
            l0dist = mctsInstance.l0Dist(mctsInstance.rootIndex)
            percent = mctsInstance.diffPercent(mctsInstance.rootIndex)
            diffs = mctsInstance.diffImage(mctsInstance.rootIndex)
            print("L2 distance %s" % l2dist)
            print("L1 distance %s" % l1dist)
            print("L0 distance %s" % l0dist)
            print("manipulated percentage distance %s" % percent)
            print("manipulated dimensions %s" % diffs)

            start_time_level = time.time()
            runningTime_level = 0
            childTerminated = False
            currentBest = eta[1]
            while runningTime_level <= MCTS_level_maximal_time:
                # Here are three steps for MCTS
                (leafNode, availableActions) = mctsInstance.treeTraversal(mctsInstance.rootIndex)
                newNodes = mctsInstance.initialiseExplorationNode(leafNode, availableActions)
                for node in newNodes:
                    (childTerminated, value) = mctsInstance.sampling(node, availableActions)
                    mctsInstance.backPropagation(node, value)
                runningTime_level = time.time() - start_time_level
                if currentBest > mctsInstance.bestCase[0]: 
                    print("best possible distance up to now is %s" % (str(mctsInstance.bestCase[0])))
                    currentBest = mctsInstance.bestCase[0]
            bestChild = mctsInstance.bestChild(mctsInstance.rootIndex)
            # pick the current best move to take  
            mctsInstance.makeOneMove(bestChild)

            image1 = mctsInstance.applyManipulation(mctsInstance.manipulation[mctsInstance.rootIndex])
            diffs = mctsInstance.diffImage(mctsInstance.rootIndex)
            path0 = "%s_pic/%s_temp_%s.png" % (dataSetName, image_index, len(diffs))
            NN.save_input(image1, path0)
            (newClass, newConfident) = NN.predict(image1)
            print("confidence: %s" % newConfident)

            # break if we found that one of the children is a misclassification
            if childTerminated is True:
                break

            # store the current best
            (_, bestManipulation) = mctsInstance.bestCase
            image1 = mctsInstance.applyManipulation(bestManipulation)
            path0 = "%s_pic/%s_currentBest.png" % (dataSetName, image_index)
            NN.save_input(image1, path0)

            numberOfMoves += 1
            runningTime_all = time.time() - start_time_all

        (_, bestManipulation) = mctsInstance.bestCase

        image1 = mctsInstance.applyManipulation(bestManipulation)
        (newClass, newConfident) = NN.predict(image1)
        newClassStr = NN.get_label(int(newClass))

        if newClass != label:
            path0 = "%s_pic/%s_%s_modified_into_%s_with_confidence_%s.png" % (
                dataSetName, image_index, origClassStr, newClassStr, newConfident)
            NN.save_input(image1, path0)
            path0 = "%s_pic/%s_diff.png" % (dataSetName, image_index)
            NN.save_input(np.subtract(image, image1), path0)
            print("\nfound an adversary image within pre-specified bounded computational resource. "
                  "The following is its information: ")
            print("difference between images: %s" % (diffImage(image, image1)))

            print("number of adversarial examples found: %s" % mctsInstance.numAdv)

            l2dist = l2Distance(mctsInstance.image, image1)
            l1dist = l1Distance(mctsInstance.image, image1)
            l0dist = l0Distance(mctsInstance.image, image1)
            percent = diffPercent(mctsInstance.image, image1)
            print("L2 distance %s" % l2dist)
            print("L1 distance %s" % l1dist)
            print("L0 distance %s" % l0dist)
            print("manipulated percentage distance %s" % percent)
            print("class is changed into '%s' with confidence %s\n" % (newClassStr, newConfident))

            return time.time() - start_time_all, newConfident, percent, l2dist, l1dist, l0dist, 0

        else:
            print("\nfailed to find an adversary image within pre-specified bounded computational resource. ")
            return 0, 0, 0, 0, 0, 0, 0


'''
