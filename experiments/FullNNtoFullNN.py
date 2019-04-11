'''
This expeirment is structured as follows:
1. Train 2 NNs with same architecture (classic 32bits CNNs)
2. Load samples that are correctly classified by both NNs (accuracies are 100% on these samples)
3. Craft adversarial samples for first NN out of samples from step 2.
4. Evaluate first and second NN on the samples from step 3.
'''