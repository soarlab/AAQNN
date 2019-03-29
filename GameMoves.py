#!/usr/bin/env python

"""
author: Xiaowei Huang

"""

from FeatureExtraction import *


############################################################
#
#  initialise possible moves for a two-player game
#
################################################################


class GameMoves:

    def __init__(self, data_set, model, image, tau):
        self.data_set = data_set
        self.model = model
        self.image = image
        self.tau = tau

        feature_extraction = FeatureExtraction(pattern='grey-box')
        kps = feature_extraction.get_key_points(self.image, num_partition=10)
        partitions = feature_extraction.get_partitions(self.image, self.model, num_partition=10)

        image1 = copy.deepcopy(self.image)

        actions = dict()
        actions[0] = kps
        s = 1
        kp2 = []

        if len(image1.shape) == 2:
            image0 = np.zeros(image1.shape)
        else:
            image0 = np.zeros(image1.shape[:2])

        # to compute a partition of the pixels, for an image classification task 
        # partitions = self.getPartition(image1, kps)
        print("The pixels are partitioned with respect to keypoints.")

        # construct moves according to the obtained the partitions 
        num_of_manipulations = 0
        for k, blocks in partitions.items():
            all_atomic_manipulations = []

            for i in range(len(blocks)):
                x = blocks[i][0]
                y = blocks[i][1]

                (_, _, chl) = image1.shape

                # + tau 
                if image0[x][y] == 0:

                    atomic_manipulation = dict()
                    for j in range(chl):
                        atomic_manipulation[(x, y, j)] = self.tau
                    all_atomic_manipulations.append(atomic_manipulation)

                    atomic_manipulation = dict()
                    for j in range(chl):
                        atomic_manipulation[(x, y, j)] = -1 * self.tau
                    all_atomic_manipulations.append(atomic_manipulation)

                image0[x][y] = 1

            # actions[k] = all_atomic_manipulations
            actions[s] = all_atomic_manipulations
            kp2.append(kps[s - 1])

            s += 1
            num_of_manipulations += len(all_atomic_manipulations)

        # index-0 keeps the keypoints, actual actions start from 1
        actions[0] = kp2
        print("the number of all manipulations initialised: %s\n" % num_of_manipulations)
        self.moves = actions

    def applyManipulation(self, image, manipulation):
        # apply a specific manipulation to have a manipulated input
        image1 = copy.deepcopy(image)
        maxVal = np.max(image1)
        minVal = np.min(image1)
        for elt in list(manipulation.keys()):
            (fst, snd, thd) = elt
            image1[fst][snd][thd] += manipulation[elt]
            if image1[fst][snd][thd] < minVal:
                image1[fst][snd][thd] = minVal
            elif image1[fst][snd][thd] > maxVal:
                image1[fst][snd][thd] = maxVal
        return image1
