import pandas as pd
import networkx as nx
import os
import re
PATH = './Dataset'


def traverseDataset(path, depth):
    global G
    dir_list = os.listdir(path)
    # print(dir_list)
    for file in dir_list:
        if (os.path.isfile(path+'/'+file)):

            if (re.match('^0x[a-fA-F0-9]{40}\.csv$', file)):
                df = pd.read_csv(path+"/"+file)
                df = df[df.isError == 0]
                H = nx.from_pandas_edgelist(df, source='From', target='To', edge_attr=[
                                            'TxHash', 'TimeStamp', 'BlockHeight', 'Value', 'isError', 'Input', 'ContractAddress'], create_using=nx.DiGraph())

                G = nx.compose(H, G)
                # print(G.edges.data())

        elif (os.path.isdir(path+"/"+file)):
            # print("pTH"+path)
            traverseDataset(path+"/"+file, depth+1)


G = nx.DiGraph()
traverseDataset(PATH, 0)
print(G)
