# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 00:51:13 2020
Graph Theory to generate features for train, validation, and test set
Data from test_node_from_train.txt this contain 4682341 edges and 1978709 nodes
@author: Derek
Idea from: https://towardsdatascience.com/learning-in-graphs-with-python-part-3-8d5513eef62d
Code from: https://medium.com/@vgnshiyer/link-prediction-in-a-social-network-df230c3d85e6
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

t1 = time.time()

# %%

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

#test_graph never be used
#test_graph=nx.read_edgelist('data/test_pos_after_eda.csv',delimiter=',',create_using=nx.DiGraph(),nodetype=int)




# %% Feature Engineering

#for followees
def jaccard_for_followees(a,b):
    try:
        if len(set(train_graph.successors(a))) == 0  | len(set(train_graph.successors(b))) == 0:
            return 0
        sim = (len(set(train_graph.successors(a)).intersection(set(train_graph.successors(b)))))/\
                                    (len(set(train_graph.successors(a)).union(set(train_graph.successors(b)))))
    except:
        return 0
    return sim
#for followers
def jaccard_for_followers(a,b):
    try:
        if len(set(train_graph.predecessors(a))) == 0  | len(set(train_graph.predecessors(b))) == 0:
            return 0
        sim = (len(set(train_graph.predecessors(a)).intersection(set(train_graph.predecessors(b)))))/\
                                 (len(set(train_graph.predecessors(a)).union(set(train_graph.predecessors(b)))))
        return sim
    except:
        return 0

#for followees
def cosine_for_followees(a,b):
    try:
        if len(set(train_graph.successors(a))) == 0  | len(set(train_graph.successors(b))) == 0:
            return 0
        sim = (len(set(train_graph.successors(a)).intersection(set(train_graph.successors(b)))))/\
                                    (math.sqrt(len(set(train_graph.successors(a)))*len((set(train_graph.successors(b))))))
        return sim
    except:
        return 0
def cosine_for_followers(a,b):
    try:
        
        if len(set(train_graph.predecessors(a))) == 0  | len(set(train_graph.predecessors(b))) == 0:
            return 0
        sim = (len(set(train_graph.predecessors(a)).intersection(set(train_graph.predecessors(b)))))/\
                                     (math.sqrt(len(set(train_graph.predecessors(a))))*(len(set(train_graph.predecessors(b)))))
        return sim
    except:
        return 0

pr = nx.pagerank(train_graph, alpha=0.85)
##pickle.dump(pr,open('data/page_rank.p','wb'))
mean_pr=float(sum(pr.values())) / len(pr)


def compute_shortest_path_length(a,b):
    p=-1
    try:
        if train_graph.has_edge(a,b):
            train_graph.remove_edge(a,b)
            p= nx.shortest_path_length(train_graph,source=a,target=b)
            train_graph.add_edge(a,b)
        else:
            p= nx.shortest_path_length(train_graph,source=a,target=b)
        return p
    except:
        return -1
    
#getting weakly connected edges from graph 
wcc=list(nx.weakly_connected_components(train_graph))
def belongs_to_same_wcc(a,b):
    index = []
    if train_graph.has_edge(b,a):
        return 1
    if train_graph.has_edge(a,b):
            for i in wcc:
                if a in i:
                    index= i
                    break
            if (b in index):
                train_graph.remove_edge(a,b)
                if compute_shortest_path_length(a,b)==-1:
                    train_graph.add_edge(a,b)
                    return 0
                else:
                    train_graph.add_edge(a,b)
                    return 1
            else:
                return 0
    else:
            for i in wcc:
                if a in i:
                    index= i
                    break
            if(b in index):
                return 1
            else:
                return 0
            
def calc_adar_in(a,b):
    sum=0
    try:
        n=list(set(train_graph.successors(a)).intersection(set(train_graph.successors(b))))
        if len(n)!=0:
            for i in n:
                sum=sum+(1/np.log10(len(list(train_graph.predecessors(i)))))
            return sum
        else:
            return 0
    except:
        return 0
    
    
def follows_back(a,b):
    if train_graph.has_edge(b,a):
        return 1
    else:
        return 0
    
    
katz = nx.katz.katz_centrality(train_graph,alpha=0.005,beta=1)
mean_katz = float(sum(katz.values())) / len(katz)

hits = nx.hits(train_graph, max_iter=100, tol=1e-08, nstart=None, normalized=True)


# %% computing the above features for our sampled data
df_final_train = pd.read_csv("data/train_after_eda.csv",names=['source_node', 'destination_node'])
df_final_test = pd.read_csv("data/test_after_eda.csv",names=['source_node', 'destination_node'])
df_final_train['indicator_link'] = y_train
df_final_test['indicator_link'] = y_test


df_final_train = df_final_train.sample(n=2000000, axis=0, replace=False)
df_final_test = df_final_test.sample(n=500000, axis=0, replace=False)


start_time = time.time()
#mapping jaccrd followers to train and test data
df_final_train['jaccard_followers'] = df_final_train.apply(lambda row:
										jaccard_for_followers(row['source_node'],row['destination_node']),axis=1)
df_final_test['jaccard_followers'] = df_final_test.apply(lambda row:
										jaccard_for_followers(row['source_node'],row['destination_node']),axis=1)

#mapping jaccrd followees to train and test data
df_final_train['jaccard_followees'] = df_final_train.apply(lambda row:
										jaccard_for_followees(row['source_node'],row['destination_node']),axis=1)
df_final_test['jaccard_followees'] = df_final_test.apply(lambda row:
										jaccard_for_followees(row['source_node'],row['destination_node']),axis=1)


	#mapping jaccrd followers to train and test data
df_final_train['cosine_followers'] = df_final_train.apply(lambda row:
										cosine_for_followers(row['source_node'],row['destination_node']),axis=1)
df_final_test['cosine_followers'] = df_final_test.apply(lambda row:
										cosine_for_followers(row['source_node'],row['destination_node']),axis=1)

#mapping jaccrd followees to train and test data
df_final_train['cosine_followees'] = df_final_train.apply(lambda row:
										cosine_for_followees(row['source_node'],row['destination_node']),axis=1)
df_final_test['cosine_followees'] = df_final_test.apply(lambda row:
										cosine_for_followees(row['source_node'],row['destination_node']),axis=1)
print("--- %s seconds ---" % (time.time() - start_time))

# %% compute number of followers, followees for both source and destination and inter followers and followees between them.
def compute_features_stage1(df_final):
    #calculating no of followers followees for source and destination
    #calculating intersection of followers and followees for source and destination
    num_followers_s=[]
    num_followees_s=[]
    num_followers_d=[]
    num_followees_d=[]
    inter_followers=[]
    inter_followees=[]
    for i,row in df_final.iterrows():
        try:
            s1=set(train_graph.predecessors(row['source_node']))
            s2=set(train_graph.successors(row['source_node']))
        except:
            s1 = set()
            s2 = set()
        try:
            d1=set(train_graph.predecessors(row['destination_node']))
            d2=set(train_graph.successors(row['destination_node']))
        except:
            d1 = set()
            d2 = set()
        num_followers_s.append(len(s1))
        num_followees_s.append(len(s2))

        num_followers_d.append(len(d1))
        num_followees_d.append(len(d2))

        inter_followers.append(len(s1.intersection(d1)))
        inter_followees.append(len(s2.intersection(d2)))
    
    return num_followers_s, num_followers_d, num_followees_s, num_followees_d, inter_followers, inter_followees
df_final_train['num_followers_s'], df_final_train['num_followers_d'], \
df_final_train['num_followees_s'], df_final_train['num_followees_d'], \
df_final_train['inter_followers'], df_final_train['inter_followees']= compute_features_stage1(df_final_train)
df_final_test['num_followers_s'], df_final_test['num_followers_d'], \
df_final_test['num_followees_s'], df_final_test['num_followees_d'], \
df_final_test['inter_followers'], df_final_test['inter_followees']= compute_features_stage1(df_final_test)

# %%
#mapping adar index on train
df_final_train['adar_index'] = df_final_train.apply(lambda row: calc_adar_in(row['source_node'],row['destination_node']),axis=1)
#mapping adar index on test
df_final_test['adar_index'] = df_final_test.apply(lambda row: calc_adar_in(row['source_node'],row['destination_node']),axis=1)

#--------------------------------------------------------------------------------------------------------
#mapping followback or not on train
df_final_train['follows_back'] = df_final_train.apply(lambda row: follows_back(row['source_node'],row['destination_node']),axis=1)

#mapping followback or not on test
df_final_test['follows_back'] = df_final_test.apply(lambda row: follows_back(row['source_node'],row['destination_node']),axis=1)

#--------------------------------------------------------------------------------------------------------
#mapping same component of wcc or not on train
df_final_train['same_comp'] = df_final_train.apply(lambda row: belongs_to_same_wcc(row['source_node'],row['destination_node']),axis=1)

##mapping same component of wcc or not on train
df_final_test['same_comp'] = df_final_test.apply(lambda row: belongs_to_same_wcc(row['source_node'],row['destination_node']),axis=1)

#--------------------------------------------------------------------------------------------------------
#mapping shortest path on train 
df_final_train['shortest_path'] = df_final_train.apply(lambda row: compute_shortest_path_length(row['source_node'],row['destination_node']),axis=1)
#mapping shortest path on test
df_final_test['shortest_path'] = df_final_test.apply(lambda row: compute_shortest_path_length(row['source_node'],row['destination_node']),axis=1)
df_final_train['page_rank_s'] = df_final_train.source_node.apply(lambda x:pr.get(x,mean_pr))
df_final_train['page_rank_d'] = df_final_train.destination_node.apply(lambda x:pr.get(x,mean_pr))

df_final_test['page_rank_s'] = df_final_test.source_node.apply(lambda x:pr.get(x,mean_pr))
df_final_test['page_rank_d'] = df_final_test.destination_node.apply(lambda x:pr.get(x,mean_pr))
#================================================================================

#Katz centrality score for source and destination in Train and test
#if anything not there in train graph then adding mean katz score
df_final_train['katz_s'] = df_final_train.source_node.apply(lambda x: katz.get(x,mean_katz))
df_final_train['katz_d'] = df_final_train.destination_node.apply(lambda x: katz.get(x,mean_katz))

df_final_test['katz_s'] = df_final_test.source_node.apply(lambda x: katz.get(x,mean_katz))
df_final_test['katz_d'] = df_final_test.destination_node.apply(lambda x: katz.get(x,mean_katz))
#================================================================================

#Hits algorithm score for source and destination in Train and test
#if anything not there in train graph then adding 0
df_final_train['hubs_s'] = df_final_train.source_node.apply(lambda x: hits[0].get(x,0))
df_final_train['hubs_d'] = df_final_train.destination_node.apply(lambda x: hits[0].get(x,0))

df_final_test['hubs_s'] = df_final_test.source_node.apply(lambda x: hits[0].get(x,0))
df_final_test['hubs_d'] = df_final_test.destination_node.apply(lambda x: hits[0].get(x,0))
#================================================================================

#Hits algorithm score for source and destination in Train and Test
#if anything not there in train graph then adding 0
df_final_train['authorities_s'] = df_final_train.source_node.apply(lambda x: hits[1].get(x,0))
df_final_train['authorities_d'] = df_final_train.destination_node.apply(lambda x: hits[1].get(x,0))

df_final_test['authorities_s'] = df_final_test.source_node.apply(lambda x: hits[1].get(x,0))
df_final_test['authorities_d'] = df_final_test.destination_node.apply(lambda x: hits[1].get(x,0))


# %% Generate the features of test-public dataset

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
#mapping jaccrd followers to train and test data
df_test_public['jaccard_followers'] = df_test_public.apply(lambda row:
										jaccard_for_followers(row['source_node'],row['destination_node']),axis=1)
#mapping jaccrd followees to train and test data
df_test_public['jaccard_followees'] = df_test_public.apply(lambda row:
										jaccard_for_followees(row['source_node'],row['destination_node']),axis=1)
#mapping jaccrd followers to train and test data
df_test_public['cosine_followers'] = df_test_public.apply(lambda row:
										cosine_for_followers(row['source_node'],row['destination_node']),axis=1)

#mapping jaccrd followees to train and test data
df_test_public['cosine_followees'] = df_test_public.apply(lambda row:
										cosine_for_followees(row['source_node'],row['destination_node']),axis=1)

df_test_public['num_followers_s'], df_test_public['num_followers_d'], \
df_test_public['num_followees_s'], df_test_public['num_followees_d'], \
df_test_public['inter_followers'], df_test_public['inter_followees']= compute_features_stage1(df_test_public)

#mapping adar index on train
df_test_public['adar_index'] = df_test_public.apply(lambda row: calc_adar_in(row['source_node'],row['destination_node']),axis=1)

#mapping followback or not on train
df_test_public['follows_back'] = df_test_public.apply(lambda row: follows_back(row['source_node'],row['destination_node']),axis=1)

#mapping same component of wcc or not on train
df_test_public['same_comp'] = df_test_public.apply(lambda row: belongs_to_same_wcc(row['source_node'],row['destination_node']),axis=1)

#mapping shortest path on train 
df_test_public['shortest_path'] = df_test_public.apply(lambda row: compute_shortest_path_length(row['source_node'],row['destination_node']),axis=1)

df_test_public['page_rank_s'] = df_test_public.source_node.apply(lambda x:pr.get(x,mean_pr))
df_test_public['page_rank_d'] = df_test_public.destination_node.apply(lambda x:pr.get(x,mean_pr))

#Katz centrality score for source and destination in Train and test
#if anything not there in train graph then adding mean katz score
df_test_public['katz_s'] = df_test_public.source_node.apply(lambda x: katz.get(x,mean_katz))
df_test_public['katz_d'] = df_test_public.destination_node.apply(lambda x: katz.get(x,mean_katz))

#Hits algorithm score for source and destination in Train and test
#if anything not there in train graph then adding 0
df_test_public['hubs_s'] = df_test_public.source_node.apply(lambda x: hits[0].get(x,0))
df_test_public['hubs_d'] = df_test_public.destination_node.apply(lambda x: hits[0].get(x,0))


#Hits algorithm score for source and destination in Train and Test
#if anything not there in train graph then adding 0
df_test_public['authorities_s'] = df_test_public.source_node.apply(lambda x: hits[1].get(x,0))
df_test_public['authorities_d'] = df_test_public.destination_node.apply(lambda x: hits[1].get(x,0))




t2 = time.time()
print("total time is: " + t2 - t1)



'''
# %% Model Building
y_train = df_final_train.indicator_link #ok
y_test = df_final_test.indicator_link #ok
df_final_train.drop(['source_node', 'destination_node','indicator_link'],axis=1,inplace=True)
df_final_test.drop(['source_node', 'destination_node','indicator_link'],axis=1,inplace=True)
#training the model
start_time = time.time()
from scipy.stats import randint as sp_randint
param_dist = {"n_estimators":sp_randint(105,125),
              "max_depth": sp_randint(10,15),
              "min_samples_split": sp_randint(110,190),
              "min_samples_leaf": sp_randint(25,65)}
'''