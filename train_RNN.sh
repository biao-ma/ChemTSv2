#!/bin/bash

# usage
#   ./train_RNN.sh train_RNN.yaml
#         $0           $1

STAT_FILE=/work/status.txt
WORK_PATH=/work/
echo Start train_RNN > $STAT_FILE

cp ${WORK_PATH}$1 /chemts/train_RNN.yaml

cd /chemts
python train_RNN/train_RNN.py -c train_RNN.yaml -s ${WORK_PATH}status.txt 1> /dev/stdout 2> /dev/stderr

if [ $? -eq 0 ]; then
    echo COMPLETE > $STAT_FILE
else
    echo ERROR > $STAT_FILE
fi

cp *.h5 ${WORK_PATH}
cp *.json ${WORK_PATH}
cp *.png ${WORK_PATH}

cd -
