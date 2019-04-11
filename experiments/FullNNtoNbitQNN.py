'''
This is expansion of 'FullNNto32bitQNN.py' experiment.

This expeirment is structured as follows:
1. Train 6 NNs with same architectures (32bits CNN, 2,4,8,16,32 bit QNN)
2. Load samples that are correctly classified by all NNs (accuracies are 100% on these samples)
3. Craft adversarial samples for 32 bits CNN out of samples from step 2.
4. Evaluate all networks on samples from step 3.

Except for steps above, see how does the same sample behaves across different quantization levels.
Want to see if there is number of bits after which a sample is not adversarial anymore.

First step is to have a map which saves id of a sample and list of successful attacks per quantization levels.
Second step would be to visualize this. TODO: How to visualize?
'''