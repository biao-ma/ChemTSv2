#!/bin/bash

# usage
#   ./chemts.sh chemts.yaml reward.yaml smiles N
#   $0          $1          $2          $3     $4 

STAT_FILE=/work/status.txt
WORK_PATH=/work/
echo Start ChemTS $1 > $STAT_FILE

cp ${WORK_PATH}$2 /chemts/reward.yaml

# copy private models
cp ${WORK_PATH}*.json /chemts/RNN-model
cp ${WORK_PATH}*.h5 /chemts/RNN-model

cd /chemts
# SMILES replace
python ligand_design/smiles_replace.py ${WORK_PATH}status.txt $3 $4
if [ ! -e "${WORK_PATH}converted.smi" ]
then
  exit 1
fi
converted=$(<${WORK_PATH}converted.smi)
echo "$converted"
# rm converted.smi

python ligand_design/mcts_ligand_main.py -c ${WORK_PATH}$1 -o ${WORK_PATH}generated.sdf \
-s ${WORK_PATH}status.txt --csv ${WORK_PATH}generated.csv $converted 1> /dev/stdout 2> /dev/stderr

if [ $? -eq 0 ]; then
    echo COMPLETE > $STAT_FILE
else
    echo ERROR > $STAT_FILE
    cd -
    exit 1
fi

cd -
