import csv
import json
import pickle
import pandas as pd
import networkx as nx
import os
import re
import matplotlib.pyplot as plt
from node2vec import Node2Vec

labels_dict = {}
phishing_total = set()
with open("combinedaddresses.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        phishing_total.add(row[0])

print(len(phishing_total))


def traverseDataset(path):
    global G
    dir_list = os.listdir(path)
    # print(dir_list)
    for file in dir_list:
        curr_path = path+'/'+file
        if (os.path.isfile(curr_path)):
            if (re.match('^0x[a-fA-F0-9]{40}\.csv$', file)):
               # print(curr_path)
                df = pd.read_csv(curr_path)
                df = df[df.isError == 0]
                H = nx.from_pandas_edgelist(df, source='From', target='To', edge_attr=[
                                            'TxHash', 'TimeStamp', 'BlockHeight', 'Value', 'isError', 'Input', 'ContractAddress'], create_using=nx.DiGraph())
                G = nx.compose(H, G)
        elif (os.path.isdir(curr_path)):
            # print("pTH"+path)
            node = os.path.splitext(os.path.basename(curr_path))[0]
            traverseDataset(curr_path)


G = nx.DiGraph()

with open('config.json', 'r') as f:
    config = json.load(f)

traverseDataset(config['DATASET_PATH'])
# print(phishing_total)
# print('0x0eF2724a6D8be9f72f3d35b62E1e8a37CEaF721e' in phishing_total)
for node in G.nodes():
    if node in phishing_total:
        print(node)
        labels_dict[node] = 1
    else:
        labels_dict[node] = 0

with open('labels.pickle', 'wb') as handle:
    pickle.dump(labels_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
# print(labels_dict)
nx.set_node_attributes(G, labels_dict, "label")
# print(G.nodes)


# print(G.nodes.data())

# sparse_adj_matrix = nx.to_scipy_sparse_array(G)
# print(sparse_adj_matrix)
# saving graph created above in gexf format
# nx.write_gexf(G, "graph_visual.gexf")


# # Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
# node2vec = Node2Vec(G, dimensions=64, walk_length=30,
#                     num_walks=200, workers=4)  # Use temp_folder for big graphs

# # Embed nodes
# # Any keywords acceptable by gensim.Word2Vec can be passed, `dimensions` and `workers` are automatically passed (from the Node2Vec constructor)
# model = node2vec.fit(window=10, min_count=1, batch_words=4)
