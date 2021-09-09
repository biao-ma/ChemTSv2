#!/bin/bash
python /work/ChemTS_dev/ligand_design/mcts_ligand_main.py -c config/chemts.yaml -o generated.sdf -s status.txt --csv generated.csv 'chemts'

