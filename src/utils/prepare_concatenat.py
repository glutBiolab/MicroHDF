"""
    The ConcatenatTestLoad.py is concatenating the feature matrices corresponding to
    each level of hierarchy (taxonomic lineage).
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
parent_path = "../"

np.set_printoptions(suppress=True)
def load_data(input_file,disease):
    last = ""
    meta_file = parent_path  + input_file + "/"+disease + "/Metadata_"+disease +".csv"
    abundance_file =parent_path  + input_file + "/"+disease + "/abundance_"+disease+ last+".csv"
    phylogenTree_file_l = parent_path  + input_file + "/"+disease + "/phylogenTree_l_"+disease+ last+".csv"
    phylogenTree_file_p = parent_path  + input_file + "/"+disease + "/phylogenTree_p_"+disease+ last+".csv"
    metadata = pd.read_csv(meta_file)
    abundance = pd.read_csv(abundance_file)
    metadata = encode_gender(metadata)
    label = pd.Categorical(metadata["disease"])
    # print("label ", label)
    metadata["disease"] = label.codes 
    # print("labels is:", metadata["disease"])
    n_sample = metadata.shape[0]
    n_feature = (abundance.shape[1] - 1) + (metadata.shape[1] - 2)

    print("sample is:", n_sample)
    print("features is:", n_feature)
    normalize_metadata = normalize_feature(metadata)
    # print(metadata.head())
    # feature integration
    label = pd.Categorical(metadata["disease"]).codes
    # pdb.set_trace()
    metadata = metadata.iloc[:, -3:].copy()
    abundance = abundance.drop(abundance.columns[0], axis=1)
    
    n_feature_2 = -1
    phylogen_l = pd.read_csv(phylogenTree_file_l)
    phylogen_p = pd.read_csv(phylogenTree_file_p)
    n_feature_2 = (phylogen_p.shape[1] - 1) + (phylogen_l.shape[1] - 1)
    phylogen_l = phylogen_l.drop(phylogen_l.columns[0], axis=1)
    phylogen_p = phylogen_p.drop(phylogen_p.columns[0], axis=1)
        # label = label[:, np.newaxis]
    label = label.reshape(-1, 1)
    data = np.concatenate((phylogen_l ,phylogen_p , abundance, metadata, label), axis=1)
    return data[:, 0:-1], data[:, -1], n_feature_2
   
def normalize_feature(metadata):
    # age, bmi nomalization
    age = pd.to_numeric(metadata['age'], errors='coerce').values
    if np.isnan(age).any():
        age[np.isnan(age)] = np.random.randint(18, 60, size=np.isnan(age).sum())
    age_scaler = MinMaxScaler()
    age_norm = age_scaler.fit_transform(age.reshape(-1, 1))
    metadata['age'] = age_norm.flatten()

    bmi = pd.to_numeric(metadata['bmi'], errors='coerce').values
    if np.isnan(bmi).any():
        bmi[np.isnan(bmi)] = np.random.uniform(16.0, 50.0, size=np.isnan(bmi).sum())
    bmi_scaler = MinMaxScaler()
    bmi_norm = bmi_scaler.fit_transform(bmi.reshape(-1, 1))
    metadata['bmi'] = bmi_norm.flatten()
    return metadata


def encode_gender(metadata):
    gender = pd.to_numeric(metadata['gender'], errors='coerce')
    if gender.isnull().any():
        gender[gender.isnull()] = np.random.randint(0, 2, size=gender.isnull().sum())
    metadata['gender'] = gender.replace({1: 'female', 0: 'male'})
    metadata['gender'] = metadata['gender'].replace({'female': 1, 'male': 0})
    return metadata








