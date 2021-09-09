'''
from subprocess import Popen, PIPE
from math import *
import random
import numpy as np
from copy import deepcopy
from types import IntType, ListType, TupleType, StringTypes
import itertools
import time
import math
import argparse
import subprocess
from keras.preprocessing import sequence
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
import sys
from rdkit.Chem import AllChem
from rdkit.Chem import MolFromSmiles, MolToSmiles
import pickle
import gzip
import networkx as nx
from rdkit.Chem import rdmolops
'''
    
from rdkit import Chem
from rdkit.Chem import Descriptors
from .filter import HashimotoFilter

hashi_filter = None 
def evaluate_score(new_compound):
    """
    Reward function that calculate score using hashimoto filter.
    
    Parameters
    -----------
    new_compound : string
        smiles string
        
    Returns
    -----------
    node_index : int
        index list(correspond to valid_compound)
    score
        score(1 or 0)
    valid_compound
        valid smiles list(delete invalid smiles from new_compound)
    """
    global hashi_filter
    if hashi_filter == None:
        hashi_filter = HashimotoFilter()

    node_index=[]
    valid_compound=[]
    score=[]
    for i in range(len(new_compound)):
        smi = new_compound[i]
        # check smiles
        mol = Chem.MolFromSmiles(smi)
        if mol:
            ret, _ = hashi_filter.filter([smi])
            if ret[0] == 0:
                print('filterd')
                continue

            node_index.append(i)
            score.append(Descriptors.MolLogP(mol))
            valid_compound.append(new_compound[i])

    return node_index,score,valid_compound
