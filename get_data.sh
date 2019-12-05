#!/bin/bash
set -x
wget http://www.stanford.edu/~jacobp2/SmallData.zip -O temp.zip 
unzip temp.zip
mv SmallData/* .
mkdir logs
mkdir results

