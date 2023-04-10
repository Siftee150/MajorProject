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

from torch_geometric_signed_directed.utils import \
    link_class_split, in_out_degree
from torch_geometric_signed_directed.nn.directed import \
    MagNet_link_prediction
from torch_geometric_signed_directed.data import \
    load_directed_real_data

labels_dict = {}
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

with open('labels.pickle', 'wb') as handle:
    pickle.dump(labels_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

nx.set_node_attributes(G, labels_dict, "y")


def train(X_real, X_img, y, edge_index,
          edge_weight, query_edges):
    model.train()
    out = model(X_real, X_img, edge_index=edge_index,
                query_edges=query_edges,
                edge_weight=edge_weight)
    loss = criterion(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_acc = accuracy_score(y.cpu(),
                               out.max(dim=1)[1].cpu())
    return loss.detach().item(), train_acc


def test(X_real, X_img, y, edge_index, edge_weight,
         query_edges):
    model.eval()
    with torch.no_grad():
        out = model(X_real, X_img, edge_index=edge_index,
                    query_edges=query_edges,
                    edge_weight=edge_weight)
    test_acc = accuracy_score(y.cpu(),
                              out.max(dim=1)[1].cpu())
    return test_acc


# print(nx.get_edge_attributes(G, 'Value'))

pyg_graph = from_networkx(
    G, group_edge_attrs=['TimeStamp', 'Value', 'BlockHeight'])
print(pyg_graph)

device = torch.device('cuda' if
                      torch.cuda.is_available() else 'cpu')

data = load_directed_real_data(dataset='webkb',
                               root='./', name='cornell').to(device)
print(data)
print(pyg_graph)

# print(pyg_graph['TxHash'])
model = MagNet_link_prediction(q=0.25, K=1, num_features=2,
                               hidden=16, label_dim=2).to(device)
criterion = torch.nn.NLLLoss()
link_data = link_class_split(pyg_graph, prob_val=0.01,
                             prob_test=0.01, task='direction', device=device)


for split in list(link_data.keys()):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01,
                                 weight_decay=0.0005)
    edge_index = link_data[split]['graph']
    edge_weight = link_data[split]['weights']

    query_edges = link_data[split]['train']['edges']
    y = link_data[split]['train']['label']
    X_real = in_out_degree(edge_index,
                           size=len(pyg_graph.y)).to(device)
    X_img = X_real.clone()
    query_val_edges = link_data[split]['val']['edges']
    y_val = link_data[split]['val']['label']
    for epoch in range(200):
        train_loss, train_acc = train(X_real,
                                      X_img, y, edge_index, edge_weight, query_edges)
        val_acc = test(X_real, X_img, y_val,
                       edge_index, edge_weight, query_val_edges)
        print(f'Split: {split:02d}, Epoch: {epoch:03d}, \
        Train_Loss: {train_loss:.4f}, Train_Acc: \
        {train_acc:.4f}, Val_Acc: {val_acc:.4f}')

    query_test_edges = link_data[split]['test']['edges']
    y_test = link_data[split]['test']['label']
    test_acc = test(X_real, X_img, y_test, edge_index,
                    edge_weight, query_test_edges)
    print(f'Split: {split:02d}, Test_Acc: {test_acc:.4f}')
    model.reset_parameters()


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
