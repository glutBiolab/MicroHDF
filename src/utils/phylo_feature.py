import sys
import os
from sklearn import preprocessing
from numpy import *
import pandas as pd
import numpy as np
from ete3 import NCBITaxa

DICT = {}
for line in open("../../data_set/Cirrhosis/abundance.tsv"):
    split = line.split("\t")
    otu = split[0].split("s__")
    if len(otu) > 1:
        otu = otu[1]
    else:
        otu = otu[0].split("g__")[1]
        # continue
    otu = otu.split("_noname")[0].split("_unclassified")[0].replace("_", " ").replace("XIII", "XIII.").strip()
    abun = np.array(split[1:], dtype=float)
    if otu in DICT:
        DICT[otu] = np.add(DICT[otu], abun)
    else:
        DICT[otu] = abun
otus = DICT.keys()
abuns = np.vstack(DICT.values()).T
# print(len(otus))
X = pd.DataFrame(data=abuns, columns=otus)
# print(X)
raw_name=X.columns.values.tolist()
ncbi = NCBITaxa()
print(raw_name)
print(len(raw_name))
print('-----------------------')
raw_id = ncbi.get_name_translator(raw_name)
raw_id = [str(i[0]) for i in list(raw_id.values())]
# print(raw_id)
# print("raw_id is type:", type(raw_id))
raw_id_dict = {}
for k, v in ncbi.get_name_translator(raw_name).items():
    if len(v) > 0:
        raw_id_dict[k] = v[0]

raw_name_new = []
raw_id_new = {}
for name in raw_name:
    if name in raw_id_dict:
        identifier = raw_id_dict[name]
        raw_name_new.append(name)
        raw_id_new[name] = identifier

raw_name = raw_name_new
raw_id = raw_id_new
# print(raw_id)
# print(raw_name)
# print(len(raw_name))
raw_id_list = list(raw_id.values())
# print(raw_id_list)
name_index = [X.columns.get_loc(name) for name in raw_name]

abuns_new = abuns[:, name_index]
X = pd.DataFrame(data=abuns_new, columns=raw_id_list)
X.columns = X.columns.astype(str)
print(type(X))
print(X)
print(X.columns)
"""使用 Newick 获得系统发育树，通过使用ETE3中提供的方法产生 PhyIoT，因为PhyIoT它不是免费的"""
# 使用get_topology方法检索指定的NCBI ID的分类树
tree = ncbi.get_topology(raw_id_list)
# print (tree.get_ascii(attributes=["taxid"]))

order = []
num = 1
for node in tree.traverse(strategy='levelorder'):
    if node.is_leaf():
        order.append(node.name)
# print(order)
postorder = []
num = 1
for node in tree.traverse(strategy='postorder'):
    if node.is_leaf():
        postorder.append(node.name)
# print(postorder)
# print(X.columns)
temp = []
for i in order:
    if i in X.columns:
        temp.append(i)

order = temp
# print(order)
temp1 = []
for i in postorder:
    if i in X.columns:
        temp1.append(i)

postorder  = temp1
# print(temp1)
known_Xl=X[order]
print(known_Xl)
known_Xp=X[postorder]
# print(known_Xl)
known_Xl.to_csv('phylogenTree_l_Cirrhosis.csv')
known_Xp.to_csv('phylogenTree_p_Cirrhosis.csv')