from mcts_ligand import mcts_ligand
import yaml
import pandas as pd
from rdkit import Chem
import os
import sys
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# リストになっている設定値を新しいDictに格納する
def get_list_config(conf):
    list_conf = {}
    for key in conf.keys():
        # print("{0}: {1} {2}".format(key, conf[key], type(conf[key])))
        if(key == "models"):
            if(len(conf["models"]) > 1):    # modelsの要素数が2以上の時にリストであると判断する
                list_conf.setdefault(key, conf[key])
            else:
                conf["models"] = conf["models"][0]  # modelsの要素数が1つの時はそのまま1つ目（index:0）の要素をconfに上書きする
        elif(key != "models" and type(conf[key]) is list):
            if(len(conf[key]) == 1):    # リストだが要素は1つの場合
                conf[key] = conf[key][0]      # 要素数が1つの時はそのまま1つ目（index:0）の要素をconfに上書きする
            else:
                list_conf.setdefault(key, conf[key])
    return list_conf

# リストになっている設定値の各要素数がn_treesと一致しているか確認
def check_num_list(list_conf, n_trees):
    for key in list_conf.keys():
        if(key == "models" and len(list_conf["models"]) != n_trees):
            return -1
        elif(len(list_conf[key]) != n_trees):
            return -2
    return 0

# その回の設定を新しいDictにセット
def set_conf(conf, list_conf, n_tree):
    new_conf = {}
    for key in conf.keys():
        if(key == "models" and type(conf["models"][0]) is list):
            '''
            modelsの1つ目の要素がリストだったらnew_confに追加
            ->get_list_config()でmodelsの要素数が1つの時はそのまま1つ目の要素をconfに上書きしているので
            その場合はconf["models"][0]はDictになっている
            DictではなくListの場合はその回に対応する要素をnew_confに追加
            '''
            new_conf["models"] = list_conf["models"][n_tree]
        elif(key != "models" and type(conf[key]) is list):    # models以外でリストになっている設定値はそのままその回に対応する要素をnew_confに追加
            new_conf[key] = list_conf[key][n_tree]
        else:
            new_conf[key] = conf[key]
    return new_conf

# DataFrameからSDFを出力
def data_frame_to_sdf(sdf_out, data_frame, list_conf=[]):
    sdf = Chem.SDWriter(sdf_out)
    for idx, row in data_frame.iterrows():
        mol = Chem.MolFromSmiles(row['compounds'])
        mol.SetProp('CHEMTS_GENERATED_SMILES', row['compounds'])
        mol.SetIntProp('CHEMTS_LOOP', row['loop'])
        mol.SetDoubleProp('CHEMTS_SCORE', row['score'])
        mol.SetIntProp('CHEMTS_DEPTH', row['depth'])
        mol.SetProp('CHEMTS_MODEL', row['model'])
        for key in list_conf.keys():
            if(key != "models"):
                mol.SetProp(key.upper(), row[key])
        sdf.write(mol)
    sdf.close()

if __name__ == "__main__":

    #argvs = sys.argv
    #argc = len(argvs)

    #init_position = None
    #init_smiles = 'chemts'
    #if argc >= 3:
    #    init_smiles = str(argvs[2])

    # コマンドラインから受け取る引数の名前を追加していく
    # ハイフンが付くと無くてもよい引数とする
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--conf', help='config yaml file', default='chemts.yaml')
    parser.add_argument('-o', '--output', help='output sdf file', default='generated.sdf')
    parser.add_argument('-s', '--status', help='status file', default='status.txt')
    parser.add_argument('--csv', help='output csv file', default='generated.csv')
    parser.add_argument('init_smiles', help='scaffold smiles')
    args = parser.parse_args()

    print('conf:', args.conf)
    print('output:', args.output)
    print('status', args.status)
    print('csv', args.csv)
    print('init_smiles:', args.init_smiles)

    f = open(str(args.conf), "r+")
    conf = yaml.load(f)
    f.close()

    # リストになっている設定値を新しいDictに格納する
    list_conf = get_list_config(conf)

    # リストになっている設定値の各要素数がn_treesと一致しているか確認
    n_trees = conf.get('n_trees', 5)
    if(check_num_list(list_conf, n_trees) != 0):
        with open(args.status, mode='w') as f:
            f.write("ERROR: List length does not match n_tree")
        print("ERROR: List length does not match n_tree")
        sys.exit(1)
    
    # create results directory
    new_dir_path = "results/"
    if not os.path.exists(new_dir_path):
        os.makedirs(new_dir_path)

    # 統合用の空DataFrameを作成
    cols = ["elapsed_time", "loop", "compounds", "score", "depth", "model"]
    sum_data_frame = pd.DataFrame(index=[], columns=cols)
    sum_data_frame = sum_data_frame.sort_index(axis=1)

    # n_treesでループ（いずれは並列処理に拡張）
    for n in range(n_trees):
        print("=== Tree - {} ========================".format(str(n + 1)))

        # 今回の木で使用する設定を新しいDictにセット
        conf_to_input = set_conf(conf, list_conf, n)
        output_files = {}
        #output_files['sdf'] = os.path.join(new_dir_path, args.output)
        #output_files['stat'] = os.path.join(new_dir_path, args.status)

        # 今回の木の結果の出力先パスを設定
        sdf_path = "{0}_{1}{2}".format(os.path.splitext(args.output)[0], str(n), os.path.splitext(args.output)[1])
        output_files['sdf'] = sdf_path
        output_files['stat'] = args.status

        # MCTSを実行
#        data_frame = mcts_ligand.MCTS(conf_to_input, output_files, args.init_smiles, n + 1)()
        data_frame = mcts_ligand.MCTS(conf_to_input, output_files,
                init_smiles=args.init_smiles, n_tree=n+1)()


        # リストになっていた設定値の内今回使用した設定値をDataFrameに追加
        for key in list_conf.keys():
            if(key != "models"):    # modelはMCTS側で追加されていたので省略
                data_frame[key] = str(list_conf[key][n])    # appendすると、int型の数値でも.0が付いてしまうのでstrで追加

        # 今回の木の結果をSDFに出力
        data_frame_to_sdf(sdf_path, data_frame, list_conf)
        # 今回の木の結果をCSVに出力
        csv_path = "{0}_{1}{2}".format(os.path.splitext(args.csv)[0], str(n), os.path.splitext(args.csv)[1])
        data_frame.to_csv(csv_path)
        #data_frame.to_csv(os.path.join(new_dir_path, args.csv))

        # 統合用DataFrameに追加
        sum_data_frame = sum_data_frame.append([data_frame], ignore_index=True)

    # 統合結果を出力
    sum_data_frame.to_csv(args.csv)
    data_frame_to_sdf(args.output, sum_data_frame, list_conf)

    smiles = list(sum_data_frame['compounds'])
    print('\n----- generated smiles (%d trees) -----' % n_trees)
    print('   unique: %d' % len(set(smiles)))
    print('    total: %d' % len(smiles))


