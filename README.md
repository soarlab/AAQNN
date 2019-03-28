# How to run
After you did both the setup of the QNN and for the attacker 'DeepGame' run the following:
```python
python launcher.py 'seed_number' 'first_image' 'last_image' 'dataset'
```
Where:

*seed_number*: set the seed for the execution. Note: if the seed does not match any of
the stored weights (previous training) the net is going to be trained from scratch.

*First Image*: index of the first image. On Cifar10, mnist max value is 9999.

*Last Image*: index of the last image. On Cifar10, mnist max value is 10000.

*dataset*: we support mnist, cifar10, fashion.

# How to start the process on CloudLab
```
$ (screen -R)
$ git clone https://github.com/soarlab/AAQNN.git && cd AAQNN && ./repo-init.sh && source venv/bin/activate
$ python launcher.py 10 0 1 mnist 
$ (ctr+a d)
$ (exit)
```

# How to run the analysis on super-computer

```python
python launcher.py 10 0 10000 mnist
```

This is how we should run the analysis. What we need to tune is only the variable
'concurrentProcesses' described in the following.

'concurrentProcesses' describes how many pairs (ImageNumber, QuantizationLevel)
are being attacked at the same time. 

Ex. (img=10,Q=2) and (img=10,Q=4) are two different pairs!

If we figure out there is space for more parallelization (ex. CPU's usage is 60%),
we can easily modify in 'executor.py' the variable 'concurrentProcesses'.

# Prerequisite
Installed pylearn2

# Install
```
python 3.6
pip3 install -r requirements.txt
```

# Test installation

```bash
python test_installation.py
```

If nothing crashes for a minute, it's all right, the installation is successful.

# Run
```
python3 launcher.py
```
