#!/bin/bash

sudo apt update
sudo apt-get install python3-pip
sudo apt-get install virtualenv

virtualenv -p python3 venv
source venv/bin/activate

git clone git://github.com/lisa-lab/pylearn2.git
cd pylearn2/
python setup.py develop
