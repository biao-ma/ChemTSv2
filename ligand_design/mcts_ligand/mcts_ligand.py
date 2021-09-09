# import群
import sys
import os
import math
import random as rand
from typing import List
import numpy as np
import time
import h5py
import importlib
import pandas as pd
import copy
import traceback

from inspect import signature
from .load_model import loaded_model
from .make_smile import create_accurate_init_position
from .add_node_type import make_input_smile, predict_smile, index_to_char, expanded_node, chem_kn_simulation_single

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from util import StatFile

INVALID_SCORE = -1024
INVALID_REWARD = -1024
MODEL_PATH = '/work/chemts/RNN-model'
# MODEL_PATH = '/chemts/RNN-model'

class Node:
    def __init__(self,
                 node_id,
                 position=None,
                 parent=None,
                 state=None,
                 isRoot=False):
        self.node_id = node_id
        self.position = position
        self.parentNode = parent
        self.childNodes = []
        self.wins = 0
        self.visits = 0
        self.depth = 0
        # self.generated_node = {}
        self.isRoot = isRoot
        self.visit_max = 100
        self.smiles = ''
        if position != None and  parent != None:
                self.smiles = parent.smiles + position

    def select_node(self, c_val):
        ucb = []
        for i in range(len(self.childNodes)):
            ucb.append(self.childNodes[i].wins / self.childNodes[i].visits +
                       c_val *
                       math.sqrt(2 * math.log(self.visits) 
                            / self.childNodes[i].visits))
        buf = ''
        for n, u in zip(self.childNodes, ucb):
            buf = buf + '%s(%.3f), ' % (n.position, u) 
        m = np.amax(ucb)
        indices = np.nonzero(ucb == m)[0]
        ind = rand.choice(indices)
        s = self.childNodes[ind]
        print(buf)
        print("  -> Select Node %d \'%s\' %.3f" % (s.node_id, s.position, m))
        return s

    def add_node(self, m, s, node_id):
        n = Node(node_id, position=m, parent=self, state=s)
        self.childNodes.append(n)
        return n

    def update_to_root(self, result):
        node = self
        while node != None:
            node.update(result)
            node = node.parentNode

    def update(self, result):
        self.visits += 1
        self.wins += result
        print('  Node %s \'%s\' updated %-.3f (wins=%.3f, visits=%d)' % \
                (self.node_id, self.smiles, result, self.wins, self.visits))

    '''

    def SetId(self, node_id):  # 使っていない
        self.node_id = node_id

    def GetId(self):  # 使っていない
        return self.node_id

    def simulation(self, state):  # 使っていない
        predicted_smile = predict_smile(model, state)
        input_smile = make_input_smile(predicted_smile)
        logp, valid_smile, all_smile = logp_calculation(input_smile)

        return logp, valid_smile, all_smile
    '''


class MCTS:
    # constructor
    def __init__(self, conf, output_files, init_smiles='chemts', n_tree=1):

        # read configurations
        self.read_configurations(conf)

        # List the words used for smiles
        self.set_vocabulary()

        # setting root 
        self.init_smiles = init_smiles         # initial SMILES
        self.node_id = 1
        # set initial state
        if self.init_smiles != 'chemts':
            # make state.position that element includes vocabulary elements
            self.init_state = create_accurate_init_position(
                self.init_smiles, self.vocabulary_list)
        else:
            self.init_state = ['&']
        # print("init_state: ", self.init_state)
        self.rootnode = Node(self.node_id, state=self.init_state, isRoot=True)

        # other settings
        self.loop = 1
        self.start_time = time.time()
        self.max_score = -1000
        self.stat = StatFile(output_files['stat'])      # status output file
        self.n_tree = n_tree    # number of trees

        # define empty variables
        self.valid_smiles_list = []
        self.valid_score_list = []
        self.loop_list = []
        self.depth_list = []
        self.elapsed_time_list = []
        self.used_models_list = []
        self.smiles_count = {}
        self.max_score_distribution = []
        self.play_counter = 0
        self.output_dict = {
            "elapsed_time": [],
            "loop": [],
            "compounds": [],
            "score": [],
            "depth": [],
            "model": []
        }

    def __call__(self):
        self.run()      # main processing

        self.stat.write('%d compounds generated. Write output files' % (len(self.output_dict["compounds"])))
        
        # output result data frame
        self.output_results()
        dataframe = pd.DataFrame(
            self.output_dict.values(), index=self.output_dict.keys()).T
        # print(dataframe)
        return dataframe

    def run(self):

        while(not self.is_complete_generation()):

            print('======= Tree %d - Loop %d - %d compounds generated =======' % (self.n_tree, self.loop, len(self.output_dict["compounds"])))
            self.stat.write('Tree %d - Loop %d - %d compounds generated' % (self.n_tree, self.loop, len(self.output_dict["compounds"])))
            
            self.node = self.rootnode
            self.state = copy.deepcopy(self.init_state)

            # selection step
            print("--- Selection ---")
            while self.node.childNodes != []:
                self.node = self.node.select_node(self.c_val)
                if self.node.position != '\n':
                    self.state.append(self.node.position)

            # expansion step
            print("--- Expansion ---")
            try:
                # List of characters that can be the next child node
                children_character_list = self.get_children_characters()
                print(children_character_list)
                # create loops which playout children_characters
                smiles_idx = len(self.valid_smiles_list)

                all_smiles_list = [None] * len(children_character_list)
                all_score_list = [None] * len(children_character_list)
                all_reward_list = [None] * len(children_character_list)

                for children_idx in range(len(children_character_list)):
                    print("{}_play start".format(str(self.play_counter)))
                    new_smiles_list = self.generate_smiles_list(
                            children_character_list[children_idx])
                    all_smiles_list[children_idx] = new_smiles_list
                    # simulation step
                    print("--- Simulation %s ---" % children_character_list[children_idx])
                    score_list = self.apply_reward_func_old(
                            new_smiles_list, smiles_idx)
                    all_score_list[children_idx] = score_list
                    all_reward_list[children_idx] = self.calculate_reward(
                            new_smiles_list, score_list)
					# debug
                    for smi, sco in zip(new_smiles_list, score_list):
                        print('%.3f, %s' % (sco, smi))
                        
                    self.play_counter = self.play_counter + 1
            except Exception as e:
                print(traceback.format_exc())
                print(self.valid_smiles_list)

            # store results
            for i in range(len(all_smiles_list)):
                self.store_results_for_dict(
                        all_smiles_list[i], all_score_list[i])
                self.update_max_score(all_score_list[i])

            # backpropation step
            print("--- Backpropagation ---")
            for i, reward in enumerate(all_reward_list):
                print("reward: ", reward)
                # no valid SMILES, score -1 and update to parent
                if reward == INVALID_REWARD:
                    self.node.update_to_root(-1.0)
                else:
                    print("child_character: ", children_character_list[i])
                    child_node = self.add_child_node(
                            children_character_list[i], reward)
                    child_node.update_to_root(reward)

            if len(all_reward_list) == 0:
                self.node.update_to_root(-1.0)
                
            # switch RNN model
            # self.switch_RNN_model()

            
            self.loop = self.loop + 1
        print("smiles_count:", self.smiles_count)

    def is_complete_generation(self):
        judgment = False
        if time.time() >= self.run_time:
            judgment = True
        if self.loop_max > 0 and self.loop > self.loop_max:
            judgment = True
        if self.cnt_max > 0 and len(self.valid_smiles_list) > self.cnt_max:
            judgment = True
        return judgment

    def read_configurations(self, conf):
        # models
        self.models = conf.get('models')
        self.model_list = []
        self.model_name_list = []
        for one_model in self.models:
            self.model_list.append(
                    loaded_model(one_model['json'], one_model['weight']))
            self.model_name_list.append(one_model['name'])
        self.models_length = len(self.models)
        self.models_index = 0
        self.model = self.model_list[self.models_index]
        self.model_name = self.model_name_list[self.models_index]

        # model switching relationships
        self.model_switch_algorithm = conf.get('model_switch_algorithm', 3)
        self.switch_model_count = 100
        if self.model_switch_algorithm == 3:
            self.switch_model_count = conf.get('model_switch_loop_count', 5)

        # reward function
        self.reward_func_module_name = conf.get('reward_func_module', 'reward_funcs')
        self.reward_func_name = conf.get('reward_func', 'check_node_type')
        self.import_module = importlib.import_module(self.reward_func_module_name)
        self.reward_func = getattr(self.import_module, self.reward_func_name)
        self.count_of_reduction_ucb1 = conf.get('count_of_reduction_ucb1', 3)
        self.print_reward_counters = None
        try:
            self.print_reward_counters = getattr(self.import_module, 'print_counters')
        except:
            print(self.reward_func_module_name, 'is not have print_counters() method')

        # other parameters required for generation
        self.playout_count = conf.get('playout_count', 5)   # playout_count
        self.c_val = conf.get('c_val', 0.8)
        self.loop_max = conf.get('loop_max', 1000)
        self.time_max = conf.get('time_max', 48 * 3600)
        self.time_max = eval(self.time_max) if type(self.time_max) is str else self.time_max
        self.cnt_max = conf.get('cnt_max', 0)
        self.use_gpu = conf.get('use_gpu', 0)
        self.seed = conf.get("seed", None)
        rand.seed(self.seed)
        np.random.seed(self.seed)
        self.run_time = time.time() + self.time_max

        # setting output pass
        self.new_dir_path = "results/"        
        self.filename = conf.get('models')[0]["json"].split("/")[-1].split(".")[0]

        print('========== display configuration ==========')
        print('reward_func_module = ', self.reward_func_module_name)
        print('reward_func = ', self.reward_func_name)
        print('C_value = ', self.c_val)
        print('count_of_reduction_ucb1 = ', self.count_of_reduction_ucb1)
        # print('model_switch_algorithm = ', self.model_switch_algorithm)
        # print('model_switch_loop_count = ', self.switch_model_count)
        print('loop_max = ', self.loop_max)
        print('time_max = ', self.time_max)
        print('cnt_max = ', self.cnt_max)
        print('use_gpu = ', self.use_gpu)
        print('playout_count = ', self.playout_count)
        print('seed = ', self.seed)
        print('models = ', self.models)

    def output_results(self):
        """
        Create json file which result of MCTS written.
        
        Parameters
        -----------
        init_smiles : string
            input smiles string. used for naming json file
        loop : list
            list of loop number that compound was generated
        smiles : list
            list of smiles of generated compound
        score : list
            list of score of generated compound
        depth : list
            list of depth that compound was generated
        elapsed_time : list
            list of time of generated compound
        used_models_list : list
            list of models that compound was generated
        """
        import json
        # output_dict = {
        #     "elapsed_time": self.elapsed_time_list,
        #     "loop": self.loop,
        #     "compounds": self.valid_smiles_list,
        #     "score": self.valid_score_list,
        #     "depth": self.depth_list,
        #     "model": self.used_models_list
        # }
        with open(self.new_dir_path + self.filename + "_" \
                + self.reward_func_module_name + "_" + self.init_smiles \
                + '.json', 'w') as f:
            json.dump(self.output_dict, f)
        print('##### generated smiles #####')
        print('   unique: %d' % len(set(self.output_dict["compounds"])))
        print('    total: %d' % len(self.output_dict["compounds"]))
    
        try:
            self.print_reward_counters()
        except:
            print('no print_counter func')

    def set_vocabulary(self):
        # List the words used for smiles
        self.all_vocabulary_list = []
        for one_model in self.models:
            with h5py.File(
                    os.path.join(MODEL_PATH,one_model['weight']), "r") as f:
                self.all_vocabulary_list.append(f['charset'][()])
        self.vocabulary_list = self.all_vocabulary_list[self.models_index]
        self.vocabulary_list = [i.decode("utf-8") if type(i) is bytes \
                else i for i in self.vocabulary_list]

    def get_children_characters(self):
        # Index of characters that can be the next child node
        children_idx_list = expanded_node(
            self.model, self.state, self.vocabulary_list, self.seed)
        # Convert index to character
        children_character_list = index_to_char(
                children_idx_list, self.vocabulary_list)
        if '\n' in children_character_list:
                    children_character_list.remove('\n')
        return children_character_list

    def generate_smiles_list(self, child_character):
        list_of_smiles_char_list = []  #2D list: List of SMILES character lists
        for i in range(self.playout_count):
            smiles_char_list = chem_kn_simulation_single(
                    self.model, self.state, self.vocabulary_list, 
                    child_character, self.seed)
            list_of_smiles_char_list.extend(smiles_char_list)
        generate_smile = predict_smile(
                list_of_smiles_char_list, self.vocabulary_list)
        new_smiles_list = make_input_smile(generate_smile)
        return list(set(new_smiles_list))

    def apply_reward_func(self, new_smiles_list, smiles_idx):
        # check reward func signature
        sig = signature(self.reward_func)
        if 'idx' in sig.parameters.keys():
            _, score_list, _ = \
                    self.reward_func(new_smiles_list, idx=smiles_idx)
        else:
            _, score_list, _ = \
                    self.reward_func(new_smiles_list)
        return score_list

    def apply_reward_func_old(self, new_smiles_list, smiles_idx):
        # check reward func signature
        sig = signature(self.reward_func)
        if 'idx' in sig.parameters.keys():
            _, score_list, valid_smiles_list = \
                    self.reward_func(new_smiles_list, idx=smiles_idx)
        else:
            _, score_list, valid_smiles_list = \
                    self.reward_func(new_smiles_list)
        # for debug
        # for i in range(len(score_list)):
        #     print("score: {0}, {1}".format(str(score_list[i]), valid_smiles_list[i]))

        # 評価関数改変前なので、リストの数を合わせる
        combined_nums_score_list = []
        for new_smiles in new_smiles_list:
            if new_smiles in valid_smiles_list:
                idx = valid_smiles_list.index(new_smiles)
                combined_nums_score_list.append(score_list[idx])
            else:
                combined_nums_score_list.append(INVALID_SCORE)
        return combined_nums_score_list

    def calculate_reward(self, new_smiles_list, score_list):
        """if there are redundant compounds, scores are -1"""
        if score_list.count(INVALID_SCORE) != len(score_list):
            # print("score_list: ", score_list)
            max_idx = score_list.index(max(score_list))
            for idx, new_smiles in enumerate(new_smiles_list):
                if score_list[idx] == INVALID_SCORE:
                    continue
                elif (self.get_smiles_count(new_smiles) + 1) \
                        >= self.count_of_reduction_ucb1:
                    print('======= reward = -1.0 =========')
                    reward = -1.0
                else:
                    #reward = (0.8 * score_list[max_idx]) / (
                    #    1 + 0.8 * abs(score_list[max_idx]))
                    def calc_reward(x):
                        return (1 / (1 + np.exp(-x/5))) * 2 - 1
                    sco = score_list[max_idx]
                    reward = calc_reward(sco)
            return reward
        else:
            return INVALID_REWARD

    def get_smiles_count(self, valid_smiles):
        """
        Returns the number of smiles output so far.
        
        Parameters
        -----------
        valid_smiles : string
            smiles string
            
        Returns
        -----------
        num_of_smiles : int
            number of smiles output so far
        """
    
        if valid_smiles in self.smiles_count:
            return self.smiles_count[valid_smiles]
        return 0

    def get_valid_smiles_and_score_list(self, new_smiles_list, score_list):
        valid_smiles_list = []
        valid_score_list = []
        for i in range(len(new_smiles_list)):
            if score_list[i] != INVALID_SCORE:
                valid_smiles_list.append(new_smiles_list[i])
                valid_score_list.append(score_list)
        return valid_smiles_list, valid_score_list

    def store_results_for_dict(self, new_smiles_list, score_list):
        for i in range(len(new_smiles_list)):
            if score_list[i] != INVALID_SCORE:
                self.output_dict["compounds"].append(new_smiles_list[i])
                self.output_dict["score"].append(score_list[i])

        len_valid_scores = len([score for score \
                in score_list if not score == INVALID_SCORE])

        temp_loop_list = [self.loop] * len_valid_scores
        self.output_dict["loop"].extend(temp_loop_list)

        temp_depth_list = [len(self.state)] * len_valid_scores
        self.output_dict["depth"].extend(temp_depth_list)

        temp_elapsed_time = [time.time() - self.start_time] * len_valid_scores
        self.output_dict["elapsed_time"].extend(temp_elapsed_time)

        temp_used_model_list = [self.model_name] * len_valid_scores
        self.output_dict["model"].extend(temp_used_model_list)

    def update_max_score(self, score_list):
        # update max score
        #print("current maximum score", self.max_score)
        for score in score_list:
            if score >= self.max_score:
                self.max_score_distribution.append(score)
                self.max_score = score
                print("Update maximum score %.3f @Loop %d" % ( 
                        self.max_score, self.loop))
            else:
                self.max_score_distribution.append(self.max_score)

    def add_child_node(self, children_character, reward):
        self.node_id += 1
        # 子ノードを追加
        new_node = self.node.add_node(
                children_character, self.state, self.node_id)
        return new_node

    def add_smiles(self, valid_smiles):
        """
        Add smiles string with generated number to list.
        
        Parameters
        -----------
        valid_smiles : string
            smiles string
        """
        if valid_smiles in self.smiles_count:
            count = self.smiles_count.get(valid_smiles)
            count += 1
        else:
            count = 1
        self.smiles_count[valid_smiles] = count

    def switch_RNN_model(self):
        """
        Judging if meeting requirements of switching RNN model.
        Now, only algorithem of using loop count is supported.
        """
        if self.model_switch_algorithm == 3:
            is_switch_model = (self.loop % self.switch_model_count == 0)
        else:
            is_switch_model = False
        if is_switch_model:
            self.models_index = (self.models_index + 1) % self.models_length
            self.model = self.model_list[self.models_index]
            self.model_name = self.model_name_list[self.models_index]
            self.vocabulary = self.all_vocabulary_list[self.models_index]
            self.vocabulary = [i.decode("utf-8") if type(i) is bytes \
                    else i for i in self.vocabulary]
      
