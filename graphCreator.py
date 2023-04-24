import csv
import json
import pickle
import pandas as pd
import networkx as nx
import os
import re
import matplotlib.pyplot as plt
from node2vec import Node2Vec
import torch
from torch_geometric.utils.convert import from_networkx
from sklearn.metrics import accuracy_score
import torch
import urllib.request as urlrequest
import time

from torch_geometric_signed_directed.utils import \
    link_class_split, in_out_degree
from torch_geometric_signed_directed.nn.directed import \
    MagNet_link_prediction
from torch_geometric_signed_directed.data import \
    load_directed_real_data

labels_dict = {}
balance_dict = {}
phishing_total = set()
with open("combinedaddresses.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        phishing_total.add(row[0])


def traverseDataset(path):
    global G
    dir_list = os.listdir(path)
    for file in dir_list:
        curr_path = path+'/'+file
        if (os.path.isfile(curr_path)):
            if (re.match('^0x[a-fA-F0-9]{40}\.csv$', file)):
               # print(curr_path)
                df = pd.read_csv(curr_path)
                df = df[df.isError == 0]
                H = nx.from_pandas_edgelist(df, source='From', target='To', edge_attr=[
                    'TimeStamp', 'BlockHeight', 'Value'], create_using=nx.DiGraph())
                G = nx.compose(H, G)
        elif (os.path.isdir(curr_path)):
            # print("pTH"+path)
            node = os.path.splitext(os.path.basename(curr_path))[0]
            traverseDataset(curr_path)


G = nx.DiGraph()

with open('config.json', 'r') as f:
    config = json.load(f)

traverseDataset(config['DATASET_PATH'])


for node in G.nodes():
    if node in phishing_total:
        labels_dict[node] = 1
    else:
        labels_dict[node] = 0

df_node = pd.DataFrame(columns=['account', 'balance'])

# FORMING NODE_ATTRIBUTE For BALANCE_VALUE IN THIS COMMENTED CODE
# for i in range(0, len(G.nodes()), 20):
#     retry = True
#     while (retry):
#         try:
#             print(i)
#             total_nodes_arg = ""
#             list_node = list(G.nodes())
#             for node in list_node[i:i+20]:
#                 total_nodes_arg = total_nodes_arg+str(node)+","
#             balance_node_url = 'https://api.etherscan.io/api?module=account&action=balancemulti&address=' + \
#                 total_nodes_arg[:len(total_nodes_arg)-1] + \
#                 '&tag=latest&apikey='+config["API_KEY"]
#             nodes_balance = urlrequest.urlopen(balance_node_url).read()
#             json_balance_res = json.loads(nodes_balance.decode('utf8'))
#             if (json_balance_res["status"] == '1'):
#                 df = pd.json_normalize(json_balance_res["result"])
#                 df_node = pd.concat([df_node, df], axis=0, ignore_index=True)
#             retry = False
#         except:
#             retry = True
#             time.sleep(1000)
# df_node.to_csv("node-attr.csv")

df = pd.read_csv('node-attr.csv', index_col=[0])
df.set_index("account", drop=True, inplace=True)
balance_dict = df.to_dict(orient='dict')['balance']

with open('labels.pickle', 'wb') as handle:
    pickle.dump(labels_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('node_attribute.pickle', 'wb') as handle:
    pickle.dump(balance_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

nx.set_node_attributes(G, labels_dict, "y")
nx.set_node_attributes(G, balance_dict, "x")
nx.write_gexf(G, "graph_visual.gexf")

pyg_graph = from_networkx(
    G, group_edge_attrs=['TimeStamp', 'Value', 'BlockHeight'])
torch.save(pyg_graph, "pyg_graph.pickle")
print(pyg_graph)
