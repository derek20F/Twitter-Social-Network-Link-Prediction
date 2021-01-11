# Link Prediction - Who is my friend?

## Data

Each user is treated as a node. And the relationship between two nodes are treated as a link (edge).
There are totally 4682341 edges and 1978709 nodes. We sample 100K of them to train our model.
The data are provided in the tab delimited txt files. In each row, the first element is the source node, and all the following nodes are the sink nodes.

## Feature Generation

In this project, we use graph theory and the [NetworkX library](https://networkx.github.io/documentation/stable/) to generate the features.
Part of the code are extracted from https://medium.com/@vgnshiyer/link-prediction-in-a-social-network-df230c3d85e6

Different regularization is used to normalize the features.

> 1. StandardScaler
> 2. RobustScaler

Features used in this project:

> ['common_neighbours', 'shortest_path', 'ccpa_front', 'ccpa_bac k', 'ccpa_0.5', 'ccpa_0.6', 'ccpa_0.7', 'ccpa_0.8', 'ccpa_0.9', 'num_followers_s', 'num_followers_d','num_followees_s', 'num_followees_d', 'inter_followers', 'inter_followees', 'cnd']

## Algorithm

MLP, KNN, SVM, and Random Forest are used as the models to implement link prediction.

## Implement

No command line input is needed. Just directly execute the .py files.

## Result

MLP got 74% AUC on the test-public set, while SVM, KNN and RF got 60~70% AUC.

## Group & Members

**Team Name: GooseHead**

- Hailey Kim
- Chen-An Fan
- Yu-Ting Liu