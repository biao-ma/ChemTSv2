from rdkit import Chem
import subprocess
from subprocess import PIPE
import sys, os

if __name__ == "__main__":
    args = sys.argv
    workdir = os.path.dirname(args[1])
    path_converted_smi = workdir + "/converted.smi"
    if(os.path.exists(path_converted_smi)):
        os.remove(path_converted_smi)

    stat_file = args[1]

    try:
        smiles = args[2]
        mol = Chem.MolFromSmiles(smiles)
        num_heavy_atoms = mol.GetNumHeavyAtoms()
        # print("num_heavy_atoms: " + str(num_heavy_atoms))
    except:
        with open(stat_file, mode='w') as f:
            f.write("ERROR: SMILES is invalid.")
        # print("ERROR: SMILES is invalid.")
        sys.exit()

    tail_num = 0
    if(len(args) >= 4):
        tail_num = int(args[3])
    # print("tail_num: " + str(tail_num))    
    
    if(tail_num == 0 or tail_num == num_heavy_atoms):
        with open(path_converted_smi, mode='w') as f:
            f.write(smiles)
    elif(1 <= tail_num and tail_num < num_heavy_atoms):
        command = "echo '{0}' | obabel -i smi -xl {1} -O {2}".format(smiles, tail_num, path_converted_smi)   # 1始まりなのでそのまま
        proc = subprocess.run(command, shell=True, stdout=PIPE, stderr=PIPE)
    else:
        with open(stat_file, mode='w') as f:
            f.write("ERROR: Atomic number is invalid.")
        # print("ERROR: Atomic number is invalid.")
