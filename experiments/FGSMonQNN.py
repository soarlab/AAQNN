'''
This expeirment is structured as follows:
1. Train QNNs for all quantization levels
2. Load samples that are correctly classified by all the QNNs from step 1 (accuracies are 100% on these samples)
3. Run the FGSM attack
6. Evaluate the QNNs on new adversarial samples, measure L2 distances etc.
'''