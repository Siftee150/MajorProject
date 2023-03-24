import pandas as pd
import networkx as nx
import os
import re
import matplotlib.pyplot as plt
from node2vec import Node2Vec
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

# saving graph created above in gexf format
nx.write_gexf(G, "graph_visual.gexf")


# # Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
# node2vec = Node2Vec(G, dimensions=64, walk_length=30,
#                     num_walks=200, workers=4)  # Use temp_folder for big graphs

# # Embed nodes
# # Any keywords acceptable by gensim.Word2Vec can be passed, `dimensions` and `workers` are automatically passed (from the Node2Vec constructor)
# model = node2vec.fit(window=10, min_count=1, batch_words=4)
