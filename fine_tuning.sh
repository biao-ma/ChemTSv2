#!/bin/bash

# usage
#   ./fine_tuning.sh fine_tuning.yaml
#         $0           $1

STAT_FILE=/work/status.txt
WORK_PATH=/work/
echo Start fine_tuning > $STAT_FILE

cp ${WORK_PATH}$1 /chemts/fine_tuning.yaml

# copy private models
cp ${WORK_PATH}*.json /chemts/RNN-model
cp ${WORK_PATH}*.h5 /chemts/RNN-model

cd /chemts
python train_RNN/fine_tuning.py -c fine_tuning.yaml -s ${WORK_PATH}status.txt 1> /dev/stdout 2> /dev/stderr

if [ $? -eq 0 ]; then
    echo COMPLETE > $STAT_FILE
else
    echo ERROR > $STAT_FILE
fi

cp *.h5 ${WORK_PATH}
cp *.json ${WORK_PATH}

cd -
