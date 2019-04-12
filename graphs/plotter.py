import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join

DIRECTORY_PATH_1 = "results1/"
DIRECTORY_PATH_2 = "results2/"

TARGET_DATASET = "mnist"


def attack_successful(fileContentInLines):
    for line in fileContentInLines:
        if line.startswith("found an adversary image"):
            return True
    return False


def parse_introduced_perturbations(fileContentInLines):
    l0 = l1 = l2 = 0
    for line in fileContentInLines:
        if l0 == 0 and line.startswith("L0"):
            l0 = int(line.split(" ")[2])
        if l1 == 0 and line.startswith("L1"):
            l1 = float(line.split(" ")[2])
        if l2 == 0 and line.startswith("L2"):
            l2 = float(line.split(" ")[2])

    return l0, l1, l2


def print_graph(perturbations, x_axis_label, y_axis_label, plot_name):
    x = []
    y = []

    for (q1_value, q2_value) in perturbations:
        x.append(q1_value)
        y.append(q2_value)

    # fig = plt.figure()
    # fig.suptitle('Dataset used for obtaining similar boundaries as a targeted neural network', fontsize=20)
    plt.xlabel(x_axis_label, fontsize=14)
    plt.ylabel(y_axis_label, fontsize=14)

    min_perturbation = min(min(x), min(y)) * 0.9
    max_perturbation = max(max(x), max(y)) * 1.1

    fig, ax = plt.subplots()

    # plot red crosses
    ax.plot([x], [y], 'r.', markersize=1)

    # plot blue 45 degrees dashed line for orientation
    ax.plot([min_perturbation, max_perturbation], [min_perturbation, max_perturbation], color='b', marker='o', linestyle='dashed', linewidth=0.5,
            markersize=3, dashes=(2, 6))  # length of 2, space of 6

    # values on axis
    plt.axis([min_perturbation, max_perturbation, min_perturbation, max_perturbation])

    plt.xlabel(x_axis_label, fontsize=14)
    plt.ylabel(y_axis_label, fontsize=14)

    plt.savefig(plot_name + ".png", bbox_inches='tight')
    plt.close()

def get_perturbations(fileNames, direcotry_path):
    perturbations = {}

    for fileName in fileNames:
        # file name format: seed_<seed>_<dataset>_<image_id>_Wbits<q>Abits<q>.txt
        analyzed_dataset = fileName.split("_")[2]

        # skip if the result is from some other dataset
        if analyzed_dataset != TARGET_DATASET:
            continue

        imageId = int(fileName.split("_")[3])
        quantizationLevel = int(fileName.split("_")[4].split(".")[0].split("Abits")[1])

        # first time analyzing the image
        if imageId not in perturbations:
            perturbations[imageId] = {}
            for qLevel in [2, 4, 8, 16, 32, 64]:
                perturbations[imageId][qLevel] = {}

        fileContent = open(direcotry_path + "/" + fileName, "r").readlines()

        l0, l1, l2 = parse_introduced_perturbations(fileContent)
        perturbations[imageId][quantizationLevel]['L0'] = l0
        perturbations[imageId][quantizationLevel]['L1'] = l1
        perturbations[imageId][quantizationLevel]['L2'] = l2

    return perturbations

def main():
    fileNames_1 = [f for f in listdir(DIRECTORY_PATH_1) if isfile(join(DIRECTORY_PATH_1, f))]
    fileNames_2 = [f for f in listdir(DIRECTORY_PATH_2) if isfile(join(DIRECTORY_PATH_2, f))]

    # testing
    perturbations_1 = get_perturbations(fileNames_1, DIRECTORY_PATH_1)
    perturbations_2 = get_perturbations(fileNames_2, DIRECTORY_PATH_2)

    perturbations_2_2_L0 = []
    perturbations_4_4_L0 = []
    perturbations_8_8_L0 = []
    perturbations_16_16_L0 = []
    perturbations_32_32_L0 = []
    perturbations_64_64_L0 = []
    perturbations_2_2_L2 = []
    perturbations_4_4_L2 = []
    perturbations_8_8_L2 = []
    perturbations_16_16_L2 = []
    perturbations_32_32_L2 = []
    perturbations_64_64_L2 = []
    for imageId in perturbations_1.keys():
        qlevel_measure_1 = perturbations_1[imageId]
        qlevel_measure_2 = perturbations_2[imageId]

        perturbations_2_2_L0.append((qlevel_measure_1[2]['L0'], qlevel_measure_2[2]['L0']))
        perturbations_4_4_L0.append((qlevel_measure_1[4]['L0'], qlevel_measure_2[4]['L0']))
        perturbations_8_8_L0.append((qlevel_measure_1[8]['L0'], qlevel_measure_2[8]['L0']))
        perturbations_16_16_L0.append((qlevel_measure_1[16]['L0'], qlevel_measure_2[16]['L0']))
        perturbations_32_32_L0.append((qlevel_measure_1[32]['L0'], qlevel_measure_2[32]['L0']))
        perturbations_64_64_L0.append((qlevel_measure_1[64]['L0'], qlevel_measure_2[64]['L0']))

        perturbations_2_2_L2.append((qlevel_measure_1[2]['L2'], qlevel_measure_2[2]['L2']))
        perturbations_4_4_L2.append((qlevel_measure_1[4]['L2'], qlevel_measure_2[4]['L2']))
        perturbations_8_8_L2.append((qlevel_measure_1[8]['L2'], qlevel_measure_2[8]['L2']))
        perturbations_16_16_L2.append((qlevel_measure_1[16]['L2'], qlevel_measure_2[16]['L2']))
        perturbations_32_32_L2.append((qlevel_measure_1[32]['L2'], qlevel_measure_2[32]['L2']))
        perturbations_64_64_L2.append((qlevel_measure_1[64]['L2'], qlevel_measure_2[64]['L2']))

    print_graph(perturbations_2_2_L0, "L0 for 2 bits", "L0 for 2 bits", "graphs/L0_2_2_bits")
    print_graph(perturbations_4_4_L0, "L0 for 4 bits", "L0 for 4 bits", "graphs/L0_4_4_bits")
    print_graph(perturbations_8_8_L0, "L0 for 8 bits", "L0 for 8 bits", "graphs/L0_8_8_bits")
    print_graph(perturbations_16_16_L0, "L0 for 16 bits", "L0 for 16 bits", "graphs/L0_16_16_bits")
    print_graph(perturbations_32_32_L0, "L0 for 32 bits", "L0 for 32 bits", "graphs/L0_32_32_bits")
    print_graph(perturbations_64_64_L0, "L0 for 64 bits", "L0 for 64 bits", "graphs/L0_64_64_bits")

    print_graph(perturbations_2_2_L2, "L2 for 2 bits", "L2 for 2 bits", "graphs/L2_2_2_bits")
    print_graph(perturbations_4_4_L2, "L2 for 4 bits", "L2 for 4 bits", "graphs/L2_4_4_bits")
    print_graph(perturbations_8_8_L2, "L2 for 8 bits", "L2 for 8 bits", "graphs/L2_8_8_bits")
    print_graph(perturbations_16_16_L2, "L2 for 16 bits", "L2 for 16 bits", "graphs/L2_16_16_bits")
    print_graph(perturbations_32_32_L2, "L2 for 32 bits", "L2 for 32 bits", "graphs/L2_32_32_bits")
    print_graph(perturbations_64_64_L2, "L2 for 64 bits", "L2 for 64 bits", "graphs/L2_64_64_bits")

if __name__ == '__main__':
    main()