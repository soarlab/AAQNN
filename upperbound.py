from __future__ import print_function
from NeuralNetwork import *
from DataSet import *
from CompetitiveMCTS import *
from CooperativeMCTS import *


def upperbound(dataSetName, bound, tau, gameType, image_index, eta, wbits, abits, nameFile, seed):
    outF = open("results/" + nameFile, "w+")

    MCTS_all_maximal_time = 500

    NN = NeuralNetwork(dataSetName, abits, wbits, 'full-qnn', seed)
    NN.train_network_QNN()
    print("Dataset is %s." % NN.data_set)
    dataset = DataSet(dataSetName, 'testing')

    realLabel = dataset.get_True_Label(image_index)
    image = dataset.get_input(image_index)
    (label, confident) = NN.predict(image)
    predicted = NN.get_label(int(label))
    print("Prediction on input with index %s, whose class predicted is '%s' (true value %s ) and the confidence is %s."
          % (image_index, predicted, NN.get_label(int(realLabel)), confident))
    outF.write(
        "Prediction on input with index %s, whose class predicted is '%s' (true value %s ) and the confidence is %s.\n"
        % (image_index, predicted, NN.get_label(int(realLabel)), confident))
    print("the second player is %s." % gameType)

    outF.write("the second player is %s.\n" % gameType)

    if not predicted == NN.get_label(int(realLabel)):
        print("NN misclassification without attack")
        outF.write("NN misclassification without attack")
        exit(0)

    outF.flush()
    # choose between "cooperative" and "competitive"
    if gameType == 'cooperative':
        mctsInstance = MCTSCooperative(dataSetName, NN, image_index, image, tau, eta)  # , target_class)
        mctsInstance.initialiseMoves()

        start_time_all = time.time()
        runningTime_all = 0
        currentBest = eta[1]
        times = {}
        number_of_current_bests = 0
        while runningTime_all <= MCTS_all_maximal_time:
            print("Time: " + str(runningTime_all))

            # Here are three steps for MCTS
            (leafNode, availableActions) = mctsInstance.treeTraversal(mctsInstance.rootIndex)
            newNodes = mctsInstance.initialiseExplorationNode(leafNode, availableActions)
            for node in newNodes:
                (_, value) = mctsInstance.sampling(node, availableActions)
                mctsInstance.backPropagation(node, value)
            if currentBest > mctsInstance.bestCase[0]:
                print("best distance up to now is %s" % (str(mctsInstance.bestCase[0])))
                currentBest = mctsInstance.bestCase[0]
                times[number_of_current_bests] = time.time() - start_time_all
                number_of_current_bests += 1

            # store the current best
            (_, bestManipulation) = mctsInstance.bestCase
            image1 = mctsInstance.applyManipulation(bestManipulation)
            path0 = "%s_pic/%s_currentBest.png" % (dataSetName, image_index)
            NN.save_input(image1, path0)

            runningTime_all = time.time() - start_time_all

        (_, bestManipulation) = mctsInstance.bestCase

        print("the number of sampling: %s" % mctsInstance.numOfSampling)
        print("the number of adversarial examples: %s\n" % mctsInstance.numAdv)

        image1 = mctsInstance.applyManipulation(bestManipulation)
        (newClass, newConfident) = NN.predict(image1)
        newClassStr = NN.get_label(int(newClass))

        if newClass != label:  # and str(target_class)==str(NN.get_label(int(newClass))):
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

            print("time needed to obtain an adversarial sample: \n")
            outF.write("time needed to obtain an adversarial sample: \n")

            for sample_index in range(0, number_of_current_bests):
                output_line = str(sample_index) + ": " + str(times[sample_index]) + "\n"
                print(output_line)
                outF.write(output_line)

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

            outF.flush()
            outF.close()
            return time.time() - start_time_all, newConfident, percent, l2dist, l1dist, l0dist, 0

        else:
            print("\nfailed to find an adversary image within pre-specified bounded computational resource. ")
            outF.write("\nfailed to find an adversary image within pre-specified bounded computational resource. \n")
            outF.flush()
            outF.close()
            return 0, 0, 0, 0, 0, 0, 0
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