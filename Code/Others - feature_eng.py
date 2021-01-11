# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 04:14:36 2020

@author: Hailey
"""


import pandas as pd
import numpy as np
import networkx as nx
import time
from tqdm import tqdm 
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
import math

#%%

ori_data = open("train.txt").read().strip()
lines = ori_data.split("\n")

datafields = []
for line in lines:
    datafields.append(line.split("\t"))

source_node_list = []
sink_node_list = []

for row in tqdm(datafields):
    for i in range(1, len(row)):
        source_node_list.append(row[0])
        sink_node_list.append(row[i])

# Generate the data frame with sour_node, sink_node pair
positive_links_df = pd.DataFrame({'Source':source_node_list, 'Sink':sink_node_list})




#%% create a directed graph
'''
#to save the memory
# create graph
G = nx.from_pandas_edgelist(positive_links_df, "Source", "Sink", create_using=nx.DiGraph())
print(nx.info(G))
totalNodesNumber = G.number_of_nodes() #get the number of unique nodes
totalEdgesNumber = G.number_of_edges() #get the number of unique edges
'''

totalEdgesNumber = 23946602


# combine all nodes in a list
node_list = source_node_list + sink_node_list

# remove duplicate items from the list
node_list = list(dict.fromkeys(node_list))
#%% Create negative links


#the dict will contain a tuple of 2 nodes as key and the value will be 1 if the nodes are connected else -1
edges = dict()
for i in tqdm(range(len(positive_links_df))):
    ##print(positive_links_df.iloc[i,0])
    edges[positive_links_df.iloc[i,0], positive_links_df.iloc[i,1]] = 1
    ##print(edge)
    
#generate the same number of negative links as positive links
missing_edges = set([])
while (len(missing_edges)<totalEdgesNumber):
    a=random.choice(node_list)
    b=random.choice(node_list)
    tmp = edges.get((a,b),-1) #when (a,b) key not exist, return -1
    if tmp == -1 and a!=b:
        missing_edges.add((a,b)) #comment out this, if we use below try-except
        '''
        # to save memory, we can't have G
        try:
            # adding points who less likely to be friends
            if nx.shortest_path_length(G,source=a,target=b) > 2: 

                missing_edges.add((a,b))
            else:
                continue  
        except:  
                missing_edges.add((a,b))
        '''        
    else:
        continue

negative_links_df = pd.DataFrame(list(missing_edges), columns=['Source', 'Sink'])

# %% Train-Validation Split
#Spiltted data into 80-20 
#positive links and negative links seperatly
#because we need positive training data only for creating graph 
#and for feature generation
# X is the source-sink pair, y is the lable
X_train_pos, X_test_pos, y_train_pos, y_test_pos  = train_test_split(positive_links_df,np.ones(len(positive_links_df)),test_size=0.2, random_state=9)
X_train_neg, X_test_neg, y_train_neg, y_test_neg  = train_test_split(negative_links_df,np.zeros(len(negative_links_df)),test_size=0.2, random_state=9)

# %%
#removing header and saving
X_train_pos.to_csv('data/train_pos_after_eda.csv',header=False, index=False)
X_test_pos.to_csv('data/test_pos_after_eda.csv',header=False, index=False)
X_train_neg.to_csv('data/train_neg_after_eda.csv',header=False, index=False)
X_test_neg.to_csv('data/test_neg_after_eda.csv',header=False, index=False)

'''
注意，我的positive和negative的數量不同
'''
# %% Merge the negative and positive into train and test
X_train = X_train_pos.append(X_train_neg,ignore_index=True)
y_train = np.concatenate((y_train_pos,y_train_neg))
X_test = X_test_pos.append(X_test_neg,ignore_index=True)
y_test = np.concatenate((y_test_pos,y_test_neg)) 

# Save to folder
X_train.to_csv('data/train_after_eda.csv',header=False,index=False)
X_test.to_csv('data/test_after_eda.csv',header=False,index=False)
pd.DataFrame(y_train.astype(int)).to_csv('data/train_y.csv',header=False,index=False)
pd.DataFrame(y_test.astype(int)).to_csv('data/test_y.csv',header=False,index=False)

train_graph=nx.read_edgelist('data/train_pos_after_eda.csv',delimiter=',',create_using=nx.DiGraph(),nodetype=int)



# %% import data
start_time = time.time()

train_graph=nx.read_edgelist('data/train_pos_after_eda.csv',delimiter=',',create_using=nx.DiGraph(),nodetype=int)

print(nx.info(train_graph))
totalNodesNumber = train_graph.number_of_nodes() #get the number of unique nodes
totalEdgesNumber = train_graph.number_of_edges() #get the number of unique edges
print("--- %s seconds ---" % (time.time() - start_time))

# %% Feature Engineering
def compute_shortest_path_length(a,b):
    p=100
    try:
        if train_graph.has_edge(a,b):
            train_graph.remove_edge(a,b)
            p= nx.shortest_path_length(train_graph,source=a,target=b)
            train_graph.add_edge(a,b)
        else:
            p= nx.shortest_path_length(train_graph,source=a,target=b)
        return p
    except:
        return 100

def common_neighbours(a,b):
    try:
        if len(set(train_graph.successors(a))) == 0  | len(set(train_graph.successors(b))) == 0:
            return 0
        cn = (len(set(train_graph.successors(a)).intersection(set(train_graph.successors(b))))) 
        return cn
    except:
        return 0

def cnd(common_neighbours, shortest_path):
    try:
        if common_neighbours == 0:
            return 1 / shortest_path
        cnd = (common_neighbours + 1) / 2
        return cnd
    except:
        return 0
    
n = train_graph.number_of_nodes()

# %% computing the above features for our sampled data
df_final_train = pd.read_csv("data/train_after_eda.csv", names=['source_node', 'destination_node'])
df_final_train['indicator_link'] = pd.read_csv("data/train_y.csv", header = None)


# #set the sample size 
# df_final_train = df_final_train.sample(n=100000, axis=0, replace=False)
# df_final_test = df_final_test.sample(n=20000, axis=0, replace=False)
keep_df_final_train = df_final_train.copy()
keep_df_final_train.shape


df_final_train = df_final_train.sample(n=200000, axis=0, replace=False, random_state = 1)

start_time = time.time()
#mapping jaccrd followees to train and test data
df_final_train['common_neighbours'] = df_final_train.apply(lambda row:
										common_neighbours(row['source_node'],row['destination_node']),axis=1)

print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
#mapping shortest path on train 
df_final_train['shortest_path'] = df_final_train.apply(lambda row: compute_shortest_path_length(row['source_node'],row['destination_node']),axis=1)
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
df_final_train['ccpa_front'] = df_final_train['common_neighbours']
df_final_train['ccpa_back'] = n/df_final_train['shortest_path']
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
df_final_train['ccpa_0.5'] = 0.5*df_final_train['ccpa_front'] + 0.5*df_final_train['ccpa_back']
df_final_train['ccpa_0.6'] = 0.6*df_final_train['ccpa_front'] + 0.4*df_final_train['ccpa_back']
df_final_train['ccpa_0.7'] = 0.7*df_final_train['ccpa_front'] + 0.3*df_final_train['ccpa_back']
df_final_train['ccpa_0.8'] = 0.8*df_final_train['ccpa_front'] + 0.2*df_final_train['ccpa_back']
df_final_train['ccpa_0.9'] = 0.9*df_final_train['ccpa_front'] + 0.1*df_final_train['ccpa_back']

print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
df_final_train['cnd'] = df_final_train.apply(lambda row: cnd(row['common_neighbours'],row['shortest_path']),axis=1)
print("--- %s seconds ---" % (time.time() - start_time))

df_final_train.to_csv('data/new_features_train.csv',header=False,index=False)

#%% Open Test Public
test_public = open("test-public.txt").read().strip()
test_public_lines = test_public.split("\n")
test_public_lines = test_public_lines[1:]

ls_test_public = []
test_public_source_node_list = []
test_public_sink_node_list = []
test_public_id_list = []
for line in test_public_lines:
	ls_test_public.append(line.split("\t"))


for row in tqdm(ls_test_public):
    test_public_id_list.append(row[0])
    test_public_source_node_list.append(row[1])
    test_public_sink_node_list.append(row[2])
	
df_test_public = pd.DataFrame(columns=['Id', 'source_node', 'destination_node'])
df_test_public['Id'] = test_public_id_list
df_test_public['source_node'] = test_public_source_node_list
df_test_public['destination_node'] = test_public_sink_node_list

df_test_public = df_test_public.drop('Id', axis = 1)


#%% Test Public
#df_test_public = pd.read_csv("data/clean_test_public.csv", names=['source_node', 'destination_node'])

start_time = time.time()
df_test_public['common_neighbours'] = df_test_public.apply(lambda row: common_neighbours(row['source_node'],row['destination_node']),axis=1)
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
df_test_public['shortest_path'] = df_test_public.apply(lambda row: compute_shortest_path_length(row['source_node'],row['destination_node']),axis=1)
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
df_test_public['ccpa_front'] = df_test_public['common_neighbours']
df_test_public['ccpa_back'] = n/df_test_public['shortest_path']
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
df_test_public['ccpa_0.5'] = 0.5*df_test_public['ccpa_front'] + 0.5*df_test_public['ccpa_back']
df_test_public['ccpa_0.6'] = 0.6*df_test_public['ccpa_front'] + 0.4*df_test_public['ccpa_back']
df_test_public['ccpa_0.7'] = 0.7*df_test_public['ccpa_front'] + 0.3*df_test_public['ccpa_back']
df_test_public['ccpa_0.8'] = 0.8*df_test_public['ccpa_front'] + 0.2*df_test_public['ccpa_back']
df_test_public['ccpa_0.9'] = 0.9*df_test_public['ccpa_front'] + 0.1*df_test_public['ccpa_back']
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
df_test_public['cnd'] = df_test_public.apply(lambda row: cnd(row['common_neighbours'],row['shortest_path']),axis=1)
print("--- %s seconds ---" % (time.time() - start_time))

df_test_public.to_csv('data/new_features_test_public.csv',header=False,index=False)

