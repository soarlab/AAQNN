'''
This expeirment is structured as follows:
1. Train QNN and save weights
2. Load weights of QNN in normal 32bits CNN, this CNN doesn't know anything about quantization
3. Repeat steps 1 and 2 for all Q levels.
4. Load samples that are correctly classified by all CNNs (accuracies are 100% on these samples)
5. Craft adversarial samples for 32 bits CNNs out of samples from step 4.
6. Evaluate the CNNs on new adversarial samples, measure L2 distances etc.
'''