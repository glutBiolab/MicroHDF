import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.utils import resample
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import AffinityPropagation
from skbio.stats.ordination import pcoa
from skbio.diversity import beta_diversity
import math
from sklearn.cluster import KMeans


class APClustering:

    _instance = None
    _resample = None
    @classmethod  
    def get_instance(cls, **arg):  
        if cls._instance is None:  
            cls._instance = cls(arg)
        return cls._instance
    
    @classmethod
    def get_resample(cls):
        if cls._resample is None:
            return None
        return cls._resample
    
    def __init__(self, arg):
       
        self.damping = arg['damping']
        self.max_iter = arg['max_iter']
        self.convergence_iter = arg['convergence_iter']
        self.simi_type = arg['simi_type']
        self.kmeans_value = arg['kmeans_value']
        self.cluster_type = arg['cluster_type']

    def cal_simi(self, X):
        dataLen = len(X)
        simi = []
        for m in X:
            temp = []
            for n in X:
                s = -np.sqrt(np.sum((m - n) ** 2))
                temp.append(s)
            simi.append(temp)
        p = np.median(simi)
        for i in range(dataLen):
            simi[i][i] = p
        return simi
    def cal_simi_diversity(self,X):
        distance_matrix = beta_diversity('braycurtis', X)
        distance_matrix = distance_matrix.to_data_frame()
        distance_matrix.fillna(0,inplace=True)
        return distance_matrix



    def init_R(self, dataLen):
        return np.zeros((dataLen, dataLen))

    def init_A(self, dataLen):
        return np.zeros((dataLen, dataLen))

    def iter_update_R(self, dataLen, R, A, simi):
        old_r = 0
        for i in range(dataLen):
            for k in range(dataLen):
                old_r = R[i][k]
                if i != k:
                    # 取A矩阵 R矩阵中i行  j != k 的最大值
                    max1 = max([simi[i][j] + A[i][j] for j in range(dataLen) if j != k])
                    R[i][k] = (1 - self.damping) * (simi[i][k] - max1) + self.damping * old_r
                else:
                    # 取相似性矩阵中i行  j != k 的最大值 
                    max2 = max([simi[i][j] for j in range(dataLen) if j != k])
                    R[i][k] = (1 - self.damping) * (simi[i][k] - max2) + self.damping * old_r
        return R

    def iter_update_A(self, dataLen, R, A):
        old_a = 0
        for i in range(dataLen):
            for k in range(dataLen):
                old_a = A[i][k]
                if i == k:
                    A[i][k] = (1 - self.damping) * sum([max(0, R[j][k]) for j in range(dataLen) if j != k]) + self.damping * old_a
                else:
                    A[i][k] = (1 - self.damping) * min(0, R[k][k] + sum([max(0, R[j][k]) for j in range(dataLen) if j != k and j != i])) + self.damping * old_a
        return A
    

    def resample_balance_probability(self,majority_class,minority_class,flag,sucess_cluster):
        cluster_center = sucess_cluster
        cluster_dir = {}
        if len(minority_class) > (len(minority_class) * 0.75):
            n_samples = math.floor(len(minority_class) * 0.75)
        else:
            n_samples = len(minority_class)
        n_samples_resample = 5
        majority_data = {}
        minority_data = {}
        for i in range(n_samples_resample):
            resample_sample = []
            resample_sample.append(resample(majority_class, replace=False, n_samples=n_samples, random_state=0))
            majority_data[i] = np.vstack(resample_sample)
        for i in range(n_samples_resample):
            minority_data[i] = resample(minority_class, replace=True, n_samples=n_samples, random_state=0)
        balanced_X = []
        balanced_y = []
        for i in range(n_samples_resample):
            balanced_X.append(np.vstack([majority_data[i], minority_data[i]])) 
            if flag == 0:
                balanced_y.append(np.hstack([np.zeros(len(majority_data[i])), np.ones(len(minority_data[i]))]))
            else:
                balanced_y.append(np.hstack([np.ones(len(majority_data[i])), np.zeros(len(minority_data[i]))]))
        return balanced_X,balanced_y
     
    def resample_balance_rate(self,majority_class,minority_class,flag,sucess_cluster):
        # np.unique(np.argmax(R + A, axis=1))
        
        cluster_center = sucess_cluster
        cluster_c_d = {}
        cluster_dir = {}
        for k in np.unique(cluster_center):
            # print(k,":",len([ 1 for i in cluster_center if i == k]))
            cluster_dir[k] = len([ 1 for i in cluster_center if i == k])
        n_samples = math.floor(len(minority_class) * 0.75)
        n_samples_resample = 5
        rate = n_samples / len(majority_class)
        max_z,max_k  = 0 , -1
        for k,v in enumerate(cluster_dir):
            cluster_c_d[v] = [ i for i in range(len(cluster_center)) if cluster_center[i] == v]
            if max_k < cluster_dir[v]:
               max_z = v
               max_k = cluster_dir[v]
        majority_data = {}
        minority_data = {}           
        for i in range(n_samples_resample):
            z_c = 0
            resample_sample = []
            for k,v in cluster_c_d.items():
                resample_sample.append(resample(majority_class[v], replace=False, n_samples=1 if math.floor(len(v) * rate) == 0 else  math.floor(len(v) * rate), random_state=0))
                z_c += math.floor(len(v) * rate)
            if z_c < n_samples:
                resample_sample.append(resample(majority_class[cluster_c_d[max_z]], replace=True, n_samples=n_samples - z_c, random_state=0))
            majority_data[i] = np.vstack(resample_sample)
        for i in range(n_samples_resample):
            minority_data[i] = resample(minority_class, replace=True, n_samples=n_samples, random_state=0)
        balanced_X = []
        balanced_y = []
        for i in range(n_samples_resample):
            balanced_X.append(np.vstack([majority_data[i], minority_data[i]])) 
            if flag == 0:
                balanced_y.append(np.hstack([np.zeros(len(majority_data[i])), np.ones(len(minority_data[i]))]))
            else:
                balanced_y.append(np.hstack([np.ones(len(majority_data[i])), np.zeros(len(minority_data[i]))]))
        return balanced_X,balanced_y

    def use_KMeans(self,k,X):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        # 获取簇中心和标签
        # centers = kmeans.cluster_centers_
        # labels = kmeans.labels_
        return kmeans
    

    def cal_cls_center(self, dataLen, simi, R, A):
        curr_iter = 0
        curr_comp = 0
        class_cen = []
        while curr_iter < self.max_iter and curr_comp < self.convergence_iter:
            R = self.iter_update_R(dataLen, R, A, simi)
            A = self.iter_update_A(dataLen, R, A)
            # 获取中心点
            new_centers = [k for k in range(dataLen) if R[k][k] + A[k][k] > 0]
            if set(new_centers) == set(class_cen):
                curr_comp += 1
            else:
                class_cen = new_centers
                curr_comp = 0
            curr_iter += 1
        if_sucess = True
        if curr_iter >= self.max_iter or curr_comp != self.convergence_iter:
            if_sucess = False
        return class_cen,if_sucess
    
    def fit_resample(self, X, y):
        flag = 0 if len(X[y == 0]) > len(X[y == 1]) else 1
        majority_class = X[y == 0] if len(X[y == 0]) > len(X[y == 1]) else X[y == 1]
        minority_class = X[y == 1] if len(X[y == 0]) > len(X[y == 1]) else X[y == 0]
        simi = None
        if self.simi_type == 1:
            simi = self.cal_simi(majority_class)
        elif self.simi_type == 2:
            simi = self.cal_simi_diversity(majority_class)
            simi = simi.values.tolist()
        dataLen = len(majority_class)
        R = self.init_R(dataLen)
        A = self.init_A(dataLen)
        class_centers,if_sucess = self.cal_cls_center(dataLen, simi, R, A)
        sampled_majority = []
        kmeans = None
        
        # 聚类不成功时 选择kmeans聚类
        if not if_sucess:
            # pdb.set_trace()
            kmeans = self.use_KMeans(self.kmeans_value,X=majority_class)
            class_centers = kmeans.labels_
            for center in np.unique(class_centers):
                cluster_samples = [i for i in range(dataLen) if class_centers[i] == center]
                sampled_cluster = resample(majority_class[cluster_samples], replace=True, n_samples=len(minority_class), random_state=0)
                sampled_majority.append(sampled_cluster)  
        else:
            for center in class_centers:
                cluster_samples = [i for i in range(dataLen) if np.argmax(simi[i]) == center]
                if len(cluster_samples) == 0:
                    continue
                sampled_cluster = resample(majority_class[cluster_samples], replace=True, n_samples=len(minority_class), random_state=0)
                sampled_majority.append(sampled_cluster)
        # pdb.set_trace()
        sampled_majority = np.vstack(sampled_majority)
        balanced_X = np.vstack([sampled_majority, minority_class])
        if flag == 0:
            balanced_y = np.hstack([np.zeros(len(sampled_majority)), np.ones(len(minority_class))])
        else:
            balanced_y = np.hstack([np.ones(len(sampled_majority)), np.zeros(len(minority_class))])
        return balanced_X, balanced_y
    
    def quick_resampled(self,X, y,random_state=None):
        # print(X.shape)
        # print(y.shape)
        # 使用随机下采样进一步平衡数据集
        cdataLen = X.shape[0]
        X_resampled, y_resampled = self.fit_resample(X, y)
        flag = 0 if len(X_resampled[y_resampled == 0]) > len(X_resampled[y_resampled == 1]) else 1    
        majority_class = X_resampled[y_resampled == 0] if len(X_resampled[y_resampled == 0]) > len(X_resampled[y_resampled == 1]) else X_resampled[y_resampled == 1]
        minority_class = X_resampled[y_resampled == 1] if len(X_resampled[y_resampled == 0]) > len(X_resampled[y_resampled == 1]) else X_resampled[y_resampled == 0]
        balanced_y = -1
        
        if (cdataLen  != len(majority_class) + len(minority_class) ) or len(majority_class) >= len(minority_class):
            majority_class_resampled = None
            if cdataLen % 2 == 1:
                majority_class_resampled = resample(majority_class, replace=False, n_samples=cdataLen // 2 + 1, random_state=random_state)
                minority_class = resample(minority_class, replace=True, n_samples=cdataLen // 2 , random_state=random_state)
            else:
                majority_class_resampled = resample(majority_class, replace=False, n_samples=cdataLen // 2, random_state=random_state)
                minority_class = resample(minority_class, replace=True, n_samples=cdataLen // 2 , random_state=random_state)
            X_resampled = np.vstack([majority_class_resampled, minority_class])

            if flag == 0:
                balanced_y = np.hstack([np.zeros(len(majority_class_resampled)), np.ones(len(minority_class))])
            else:
                balanced_y = np.hstack([np.ones(len(majority_class_resampled)), np.zeros(len(minority_class))])
        return X_resampled,balanced_y
    
    def fit_init_resample(self, X, y):
        flag = 0 if len(X[y == 0]) > len(X[y == 1]) else 1
        majority_class = X[y == 0] if len(X[y == 0]) > len(X[y == 1]) else X[y == 1]
        minority_class = X[y == 1] if len(X[y == 0]) > len(X[y == 1]) else X[y == 0]
        simi = None
        # 欧式距离
        if self.simi_type == 1: 
            simi = self.cal_simi(majority_class)
        # braycurtis
        elif self.simi_type == 2:
            simi = self.cal_simi_diversity(majority_class)
            # DataFrame  转  列表
            simi = simi.values.tolist()
        dataLen = len(majority_class)
        R = self.init_R(dataLen)
        A = self.init_A(dataLen)
        class_centers,if_sucess = self.cal_cls_center(dataLen, simi, R, A)
        if not if_sucess:

            kmeans = self.use_KMeans(self.kmeans_value,X=majority_class)
            class_centers = kmeans.labels_
        else:
            class_centers = np.argmax(R + A, axis=1)

        # 选择采样方式
        balanced_X,balanced_y = None,None
        if self.cluster_type == 1:
            balanced_X,balanced_y = self.resample_balance_rate(majority_class,minority_class,flag,class_centers)
        else:
            balanced_X,balanced_y = self.resample_balance_probability(majority_class,minority_class,flag,class_centers)

        APClustering._resample = {'X':balanced_X,'Y':balanced_y}
    
    def resetparam(self,damping,max_iter, convergence_iter):
        self.damping = damping
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter
