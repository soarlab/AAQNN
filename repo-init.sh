#!/bin/bash

sudo apt update
sudo apt-get --yes install python3-pip
sudo apt-get --yes install virtualenv

virtualenv -p python3 venv
source venv/bin/activate
yes | pip install -r requirements.txt

git clone git://github.com/lisa-lab/pylearn2.git
cd pylearn2/
python setup.py develop
