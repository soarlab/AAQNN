import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join

DIRECTORY_PATH = "results/"

TARGET_DATASET = "mnist"


fileNames = [f for f in listdir(DIRECTORY_PATH) if isfile(join(DIRECTORY_PATH, f))]

numOfunsuccessfulAttacks = 0
numOfTotalAttacks = 0


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

    max_perturbation = max(max(x), max(y)) * 1.0

    fig, ax = plt.subplots()

    # plot red crosses
    ax.plot([x], [y], 'rx')

    # plot blue 45 degrees dashed line for orientation
    ax.plot([0, max_perturbation], [0, max_perturbation], color='b', marker='o', linestyle='dashed', linewidth=0.5,
            markersize=3, dashes=(2, 6))  # length of 2, space of 6

    # values on axis
    plt.axis([0, max_perturbation, 0, max_perturbation])

    plt.xlabel(x_axis_label, fontsize=14)
    plt.ylabel(y_axis_label, fontsize=14)

    plt.savefig(plot_name + ".png", bbox_inches='tight')
    plt.close()


perturbations = {}

for fileName in fileNames:
    # file name format: seed_<seed>_<dataset>_<image_id>_Wbits<q>Abits<q>.txt
    analyzed_dataset = fileName.split("_")[2]

    # skip if the result is from some other dataset
    if analyzed_dataset != TARGET_DATASET:
        continue

    numOfTotalAttacks += 1
    imageId = int(fileName.split("_")[3])
    quantizationLevel = int(fileName.split("_")[4].split(".")[0].split("Abits")[1])

    # first time analyzing the image
    if imageId not in perturbations:
        perturbations[imageId] = {}
        for qLevel in [2, 4, 8, 16, 32, 64]:
            perturbations[imageId][qLevel] = {}

    fileContent = open(DIRECTORY_PATH + "/" + fileName, "r").readlines()

    if not attack_successful(fileContent):
        numOfunsuccessfulAttacks += 1

    l0, l1, l2 = parse_introduced_perturbations(fileContent)
    perturbations[imageId][quantizationLevel]['L0'] = l0
    perturbations[imageId][quantizationLevel]['L1'] = l1
    perturbations[imageId][quantizationLevel]['L2'] = l2

# L0 graphs #########
#  list of tuples: (perturbation introduced for x bits, perturbation introduced for y bits)
perturbations_2_4_bits_L0 = []
perturbations_2_8_bits_L0 = []
perturbations_2_16_bits_L0 = []
perturbations_2_32_bits_L0 = []
perturbations_2_64_bits_L0 = []

perturbations_4_8_bits_L0 = []
perturbations_4_16_bits_L0 = []
perturbations_4_32_bits_L0 = []
perturbations_4_64_bits_L0 = []

perturbations_8_16_bits_L0 = []
perturbations_8_32_bits_L0 = []
perturbations_8_64_bits_L0 = []

perturbations_16_32_bits_L0 = []
perturbations_16_64_bits_L0 = []

perturbations_32_64_bits_L0 = []

for imageId in perturbations.keys():
    qlevel_measure = perturbations[imageId]

    perturbations_2_4_bits_L0.append((qlevel_measure[2]['L0'], qlevel_measure[4]['L0']))
    perturbations_2_8_bits_L0.append((qlevel_measure[2]['L0'], qlevel_measure[8]['L0']))
    perturbations_2_16_bits_L0.append((qlevel_measure[2]['L0'], qlevel_measure[16]['L0']))
    perturbations_2_32_bits_L0.append((qlevel_measure[2]['L0'], qlevel_measure[32]['L0']))
    perturbations_2_64_bits_L0.append((qlevel_measure[2]['L0'], qlevel_measure[64]['L0']))

    perturbations_4_8_bits_L0.append((qlevel_measure[4]['L0'], qlevel_measure[8]['L0']))
    perturbations_4_16_bits_L0.append((qlevel_measure[4]['L0'], qlevel_measure[16]['L0']))
    perturbations_4_32_bits_L0.append((qlevel_measure[4]['L0'], qlevel_measure[32]['L0']))
    perturbations_4_64_bits_L0.append((qlevel_measure[4]['L0'], qlevel_measure[64]['L0']))

    perturbations_8_16_bits_L0.append((qlevel_measure[8]['L0'], qlevel_measure[16]['L0']))
    perturbations_8_32_bits_L0.append((qlevel_measure[8]['L0'], qlevel_measure[32]['L0']))
    perturbations_8_64_bits_L0.append((qlevel_measure[8]['L0'], qlevel_measure[64]['L0']))

    perturbations_16_32_bits_L0.append((qlevel_measure[16]['L0'], qlevel_measure[32]['L0']))
    perturbations_16_64_bits_L0.append((qlevel_measure[16]['L0'], qlevel_measure[64]['L0']))

    perturbations_32_64_bits_L0.append((qlevel_measure[32]['L0'], qlevel_measure[64]['L0']))

print_graph(perturbations_2_4_bits_L0, "L0 for 2 bits", "L0 for 4 bits", "graphs/L0_2_4_bits")
print_graph(perturbations_2_8_bits_L0, "L0 for 2 bits", "L0 for 8 bits", "graphs/L0_2_8_bits")
print_graph(perturbations_2_16_bits_L0, "L0 for 2 bits", "L0 for 16 bits", "graphs/L0_2_16_bits")
print_graph(perturbations_2_32_bits_L0, "L0 for 2 bits", "L0 for 32 bits", "graphs/L0_2_32_bits")
print_graph(perturbations_2_64_bits_L0, "L0 for 2 bits", "L0 for 64 bits", "graphs/L0_2_64_bits")

print_graph(perturbations_4_8_bits_L0, "L0 for 4 bits", "L0 for 8 bits", "graphs/L0_4_8_bits")
print_graph(perturbations_4_16_bits_L0, "L0 for 4 bits", "L0 for 16 bits", "graphs/L0_4_16_bits")
print_graph(perturbations_4_32_bits_L0, "L0 for 4 bits", "L0 for 32 bits", "graphs/L0_4_32_bits")
print_graph(perturbations_4_64_bits_L0, "L0 for 4 bits", "L0 for 64 bits", "graphs/L0_4_64_bits")

print_graph(perturbations_8_16_bits_L0, "L0 for 8 bits", "L0 for 16 bits", "graphs/L0_8_16_bits")
print_graph(perturbations_8_32_bits_L0, "L0 for 8 bits", "L0 for 32 bits", "graphs/L0_8_32_bits")
print_graph(perturbations_8_64_bits_L0, "L0 for 8 bits", "L0 for 64 bits", "graphs/L0_8_64_bits")

print_graph(perturbations_16_32_bits_L0, "L0 for 16 bits", "L0 for 32 bits", "graphs/L0_16_32_bits")
print_graph(perturbations_16_64_bits_L0, "L0 for 16 bits", "L0 for 64 bits", "graphs/L0_16_64_bits")

print_graph(perturbations_32_64_bits_L0, "L0 for 32 bits", "L0 for 64 bits", "graphs/L0_32_64_bits")

# L2 graphs #########
#  list of tuples: (perturbation introduced for x bits, perturbation introduced for y bits)
perturbations_2_4_bits_L2 = []
perturbations_2_8_bits_L2 = []
perturbations_2_16_bits_L2 = []
perturbations_2_32_bits_L2 = []
perturbations_2_64_bits_L2 = []

perturbations_4_8_bits_L2 = []
perturbations_4_16_bits_L2 = []
perturbations_4_32_bits_L2 = []
perturbations_4_64_bits_L2 = []

perturbations_8_16_bits_L2 = []
perturbations_8_32_bits_L2 = []
perturbations_8_64_bits_L2 = []

perturbations_16_32_bits_L2 = []
perturbations_16_64_bits_L2 = []

perturbations_32_64_bits_L2 = []

for imageId in perturbations.keys():
    qlevel_measure = perturbations[imageId]

    perturbations_2_4_bits_L2.append((qlevel_measure[2]['L2'], qlevel_measure[4]['L2']))
    perturbations_2_8_bits_L2.append((qlevel_measure[2]['L2'], qlevel_measure[8]['L2']))
    perturbations_2_16_bits_L2.append((qlevel_measure[2]['L2'], qlevel_measure[16]['L2']))
    perturbations_2_32_bits_L2.append((qlevel_measure[2]['L2'], qlevel_measure[32]['L2']))
    perturbations_2_64_bits_L2.append((qlevel_measure[2]['L2'], qlevel_measure[64]['L2']))

    perturbations_4_8_bits_L2.append((qlevel_measure[4]['L2'], qlevel_measure[8]['L2']))
    perturbations_4_16_bits_L2.append((qlevel_measure[4]['L2'], qlevel_measure[16]['L2']))
    perturbations_4_32_bits_L2.append((qlevel_measure[4]['L2'], qlevel_measure[32]['L2']))
    perturbations_4_64_bits_L2.append((qlevel_measure[4]['L2'], qlevel_measure[64]['L2']))

    perturbations_8_16_bits_L2.append((qlevel_measure[8]['L2'], qlevel_measure[16]['L2']))
    perturbations_8_32_bits_L2.append((qlevel_measure[8]['L2'], qlevel_measure[32]['L2']))
    perturbations_8_64_bits_L2.append((qlevel_measure[8]['L2'], qlevel_measure[64]['L2']))

    perturbations_16_32_bits_L2.append((qlevel_measure[16]['L2'], qlevel_measure[32]['L2']))
    perturbations_16_64_bits_L2.append((qlevel_measure[16]['L2'], qlevel_measure[64]['L2']))

    perturbations_32_64_bits_L2.append((qlevel_measure[32]['L2'], qlevel_measure[64]['L2']))

print_graph(perturbations_2_4_bits_L2, "L2 for 2 bits", "L2 for 4 bits", "graphs/L2_2_4_bits")
print_graph(perturbations_2_8_bits_L2, "L2 for 2 bits", "L2 for 8 bits", "graphs/L2_2_8_bits")
print_graph(perturbations_2_16_bits_L2, "L2 for 2 bits", "L2 for 16 bits", "graphs/L2_2_16_bits")
print_graph(perturbations_2_32_bits_L2, "L2 for 2 bits", "L2 for 32 bits", "graphs/L2_2_32_bits")
print_graph(perturbations_2_64_bits_L2, "L2 for 2 bits", "L2 for 64 bits", "graphs/L2_2_64_bits")

print_graph(perturbations_4_8_bits_L2, "L2 for 4 bits", "L2 for 8 bits", "graphs/L2_2_8_bits")
print_graph(perturbations_4_16_bits_L2, "L2 for 4 bits", "L2 for 16 bits", "graphs/L2_4_16_bits")
print_graph(perturbations_4_32_bits_L2, "L2 for 4 bits", "L2 for 32 bits", "graphs/L2_4_32_bits")
print_graph(perturbations_4_64_bits_L2, "L2 for 4 bits", "L2 for 64 bits", "graphs/L2_4_64_bits")

print_graph(perturbations_8_16_bits_L2, "L2 for 8 bits", "L2 for 16 bits", "graphs/L2_8_16_bits")
print_graph(perturbations_8_32_bits_L2, "L2 for 8 bits", "L2 for 32 bits", "graphs/L2_8_32_bits")
print_graph(perturbations_8_64_bits_L2, "L2 for 8 bits", "L2 for 64 bits", "graphs/L2_8_64_bits")

print_graph(perturbations_16_32_bits_L2, "L2 for 16 bits", "L2 for 32 bits", "graphs/L2_16_32_bits")
print_graph(perturbations_16_64_bits_L2, "L2 for 16 bits", "L2 for 64 bits", "graphs/L2_16_64_bits")

print_graph(perturbations_32_64_bits_L2, "L2 for 32 bits", "L2 for 64 bits", "graphs/L2_32_64_bits")