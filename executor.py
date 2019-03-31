from __future__ import print_function
import sys
import multiprocessing
from main import process_image
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
For every image, an adversarial sample is crated for every quantization level (2, 4, .. 64).
Creation of an adversarial sample is isolated in a separate process.
It follows that for every image, there will be 6 different processes.
But for creation of an adversarial sample for the next image, it is not necessary to wait for the previous image to finish.
So the creation of adversarial samples for the next image starts in parallel with the creation of adv samples for the previous image and so on.
'''


def main():
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    print("pool size: ")
    quantization = [2, 4, 8, 16, 32, 64]

    print(sys.argv[1])
    seed = int(sys.argv[1])
    start_point = int(sys.argv[2])
    test_length = int(sys.argv[3])
    dataset_name = str(sys.argv[4])

    for sourceImage in range(start_point, test_length):
        print(sourceImage)
        index0 = sourceImage
        for q in quantization:
            pool.apply_async(process_image, [dataset_name, "ub", "cooperative", index0, "L2", 10, 1, q, q, seed])

    # wait for all jobs to finish
    pool.close()
    pool.join()
    print("all done")


if __name__=="__main__":
    main()
