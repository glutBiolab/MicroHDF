from models.gcForest import gcForest
from sklearn.model_selection import RepeatedStratifiedKFold
import numpy as np
from utils.evaluation import f1_binary,f1_micro, auc_scores

#Addition of phylogenetic tree structure data
from utils.prepare_concatenat import load_data
# from test_load_data import test_load_data
from utils.feature_selection_test import feature_select_f3,feature_select_lda
from sklearn.metrics import precision_score,f1_score,recall_score,auc,precision_recall_curve,accuracy_score, roc_curve
from models.APClustering import APClustering
import argparse
import sys
def get_config():
    config={}
    config["random_state"]=None
    config["max_layers"]=100
    config["early_stop_rounds"]=1
    config["if_stacking"]=False
    config["if_save_model"]=False
    config["train_evaluation"]=f1_binary    # f1_binary,f1_macro,f1_micro,accuracy
    config["if_resample"] = None
    config["estimator_configs"]=[]
    config["estimator_configs"].append({"n_fold":5,"type":"RandomForestClassifier","n_estimators":100,"n_jobs":-1})
    config["estimator_configs"].append({"n_fold":5,"type":"RandomForestClassifier","n_estimators":100,"n_jobs":-1})
    config["estimator_configs"].append({"n_fold":5,"type":"ExtraTreesClassifier","n_estimators":100,"n_jobs":-1})
    config["estimator_configs"].append({"n_fold":5,"type":"ExtraTreesClassifier","n_estimators":100,"n_jobs":-1})
    config["output_layer_config"]=0
    return config
def get_config1():
    config={}
    config["random_state"]=None
    config["max_layers"]=100
    config["early_stop_rounds"]=1
    config["if_stacking"]=False
    config["if_save_model"]=False
    config["train_evaluation"]=f1_micro
    config["if_resample"] = None
    config["estimator_configs"]=[]
    config["estimator_configs"].append({"n_fold":5,"type":"RandomForestClassifier","n_estimators":50,"n_jobs":-1})
    config["estimator_configs"].append({"n_fold":5,"type":"RandomForestClassifier","n_estimators":50,"n_jobs":-1})
    config["estimator_configs"].append({"n_fold":5,"type":"ExtraTreesClassifier","n_estimators":50,"n_jobs":-1})
    config["estimator_configs"].append({"n_fold":5,"type":"ExtraTreesClassifier","n_estimators":50,"n_jobs":-1})
    config["output_layer_config"]=[]
    return config
def get_config2():
    config={}
    config["random_state"]=None
    config["max_layers"]=100
    config["early_stop_rounds"]=1
    config["if_stacking"]=False
    config["if_save_model"]=False
    config["train_evaluation"]=f1_binary
    config["if_resample"] = None
    config["estimator_configs"]=[]
    config["estimator_configs"].append({"n_fold":5,"type":"RandomForestClassifier","n_estimators":50,"n_jobs":-1})
    config["estimator_configs"].append({"n_fold":5,"type":"RandomForestClassifier","n_estimators":50,"n_jobs":-1})
    config["estimator_configs"].append({"n_fold":5,"type":"ExtraTreesClassifier","n_estimators":50,"n_jobs":-1})
    config["estimator_configs"].append({"n_fold":5,"type":"ExtraTreesClassifier","n_estimators":50,"n_jobs":-1})
    config["output_layer_config"]=[]
    return config




def print_info(disease,f1s,accuracys,auprs,recalls,aucs):
    print("============training finished============")
    

    f1s=np.array(f1s)
    accs=np.array(accuracys)
    auprs=np.array(auprs)
    recalls=np.array(recalls)
    def print_mean_std(metric_name, values):
        mean = np.mean(values)
        std = np.std(values)
        print(f"{metric_name}: {mean:.4f} ± {std:.4f}")

    for recall in recalls:
        print('recall:',recall)
    print("Data:", disease)
    print_mean_std("auc", aucs)
    print_mean_std("aupr", auprs)
    print_mean_std("accuracy", accs)
    print_mean_std("recall", recalls)
    print_mean_std("f1", f1s)


def multi_channel(x, y, n_feature_2,cvfold,disease,simi_type,kmeans_value,run_cluster_resample,damping,max_iter,convergence_iter,select_feature,cluster_type):

    ap = APClustering.get_instance(damping=damping, max_iter=max_iter, convergence_iter=convergence_iter,simi_type = simi_type, kmeans_value = kmeans_value,cluster_type= cluster_type)

    skf=RepeatedStratifiedKFold(n_splits=cvfold,random_state=16,n_repeats=1)
    accuracys=[]
    aucs=[]
    f1s=[]
    auprs=[]
    recalls=[]
    i = 1
    # pdb.set_trace()
    for train_id,test_id in skf.split(x,y):
        print("============{}-th cross validation============".format(i))


        # 分别表示丰度数据和系统发育树信息
        mate_data = x[:, n_feature_2:]
        phylogen = x[:, 0:n_feature_2]
        
        mate_data_train, mate_data_test = mate_data[train_id], mate_data[test_id]
        phylogen_train, phylogen_test = phylogen[train_id], phylogen[test_id]
        print("mate_data_train", mate_data_train.shape)
        print("mate_data_test", mate_data_test.shape)

        

        y_train, y_test = y[train_id], y[test_id]
        mate_index = None
        phylogen_index = None
        #特征重排
        if select_feature == 1:
            mate_index = feature_select_f3(mate_data_train, y_train,mate_data_train.shape[1])
            phylogen_index = feature_select_f3(phylogen_train, y_train,phylogen_train.shape[1])
        else:
            mate_index = feature_select_lda(mate_data_train, y_train,mate_data_train.shape[1])
            phylogen_index = feature_select_lda(phylogen_train, y_train,phylogen_train.shape[1])
        
        # pdb.set_trace()
        # 构造优特征的数据集
        mate_data_train = mate_data_train[:, mate_index]
        phylogen_train = phylogen_train[:, phylogen_index]
        mate_data_test = mate_data_test[:, mate_index]
        phylogen_test = phylogen_test[:, phylogen_index]
        # pdb.set_trace()
        
        zzz  = mate_data_train
        
        # train gcForse for modality 1
        # 选择是否在RFs中开启训练
        y_mate = y_train
        config  = get_config1()
        if run_cluster_resample == 1:
            ap.fit_init_resample(mate_data_train,y_train)
            config['if_resample'] = True
        
        gc1 = gcForest(config)
        gc1.fit(mate_data_train, y_mate)
        mate_data_train_features = gc1.predict_proba(mate_data_train)  # shape: (n_samples, n_features1)
        mate_data_test_features = gc1.predict_proba(mate_data_test)  # shape: (n_samples, n_features1)
        
        
        # train gcForest for modality 2
        # 选择是否在RFs中开启训练
        y_phylogen = y_train
        config  = get_config2()
        if run_cluster_resample == 1:
            ap.fit_init_resample(phylogen_train,y_train)
            config['if_resample'] = True
        
        gc2 = gcForest(config)
        gc2.fit(phylogen_train, y_phylogen)
        phylogen_train_features = gc2.predict_proba(phylogen_train)  # shape: (n_samples, n_features2)
        phylogen_test_features = gc2.predict_proba(phylogen_test)  # shape: (n_samples, n_features2)

        # concatenate features from both modalities
        x_train_features = np.concatenate((mate_data_train_features, phylogen_train_features ,mate_data_train, phylogen_train),axis=1)  # shape: (n_samples, n_features1 + n_features2)
        print("x_train_feature shape:", x_train_features.shape)
        x_test_features = np.concatenate((mate_data_test_features, phylogen_test_features,mate_data_test, phylogen_test), axis=1)  # shape: (n_samples, n_features1 + n_features2)

        config = get_config()
        gc = gcForest(config)

        
       
        gc.fit(x_train_features, y_train)
        y_pred = gc.predict(x_test_features)
        y_pred_prob = gc.predict_proba(x_test_features)

        y_score = y_pred_prob[:, 1]
        y_score = []
        for item in y_pred_prob:
            y_score.append(item[1])
        y_score = np.array(y_score)  
        precision, recall, thresholds = precision_recall_curve(y_test, y_score)
        aupr = auc(recall,precision)

        # compute model output tpr and fpr, and save as file
        fpr, tpr, _ = roc_curve(y_test, y_score)

        accuracy = accuracy_score(y_test, y_pred)
        
        f1 = f1_score(y_test, y_pred,average='binary')
        recall = recall_score(y_test, y_pred,average="binary")
        precision = precision_score(y_test,y_pred)

        auc_s = auc_scores(y_test, y_pred_prob[:, 1])
          
        f1s.append(f1)
        accuracys.append(accuracy)
        aucs.append(auc_s)
        auprs.append(aupr)
        recalls.append(recall)
        # pdb.set_trace()

        with open('../logs/tpr_fpr.txt', 'a') as f:
            f.write(f"========={i}-th cross validation=========\n")
            for j in range(len(fpr)):
                f.write(f"FPR: {fpr[j]}\tTPR: {tpr[j]}\n")

        i += 1

    print_info(disease,f1s,accuracys,auprs,recalls,aucs)

    

def main():
    usage="MicroHDF -  an interpretable deep learning framework to predict host phenotypes, where a cascade layers of deep forest units is designed for handling sample class imbalance and high dimensional features."
    parser=argparse.ArgumentParser(prog="multi_channel_main.py",description=usage)
    parser.add_argument('-i','--input_file',dest='input_file',type=str,help="The directory of the input csv file.")
    parser.add_argument('-d','--disease',dest='disease',type=str,help="The name of the disease.")
    parser.add_argument('-dp','--damping',dest='damping',type=str,help="The value of damping in APCluster.(default: 0.5)")
    parser.add_argument('-mi','--max_iter',dest='max_iter',type=str,help="The value of max_iter in APCluster.(default: 50)")
    parser.add_argument('-ci','--convergence_iter',dest='convergence_iter',type=str,help="The value of convergence_iter in APCluster.(default: 25)")
    parser.add_argument('-c','--cvfold',dest='cvfold',type=str,help="The value of k in k-fold cross validation.  (default: 5)")
    parser.add_argument('-p','--simi_type',dest='simi_type',type=str,help="The value simi_type represents two methods of similarity: braycurtis and distance similarity (default: distance similarity).")
    parser.add_argument('-k','--kmeans',dest='kmeans',type=str,help="The value kmeans stands for the number of K-means clusters.(default: 5)")
    parser.add_argument('-rcr','--run_cluster_resample',dest='run_cluster_resample',type=str,help="Whether run cluster and resample process.If set to 1, then will add the process. (default: run)")
    parser.add_argument('-s','--select_feature',dest='select_feature',type=str,help="The value select_feature stands for the way features are filtered.  (default: 1)")
    parser.add_argument('-ct','--cluster_type',dest='cluster_type',type=str,help="The value cluster_type represents the sampling method: We provide two sampling methods: probabilistic and proportional.  (default: proportional)")

    args=parser.parse_args()
    input_file=args.input_file
    cvfold=int(args.cvfold) if args.cvfold !=None else 5
    disease = args.disease
    damping = float(args.damping) if args.damping !=None else 0.5
    max_iter = float(args.max_iter) if args.max_iter !=None else 50
    convergence_iter = float(args.convergence_iter) if args.convergence_iter !=None else 25
    simi_type = int(args.simi_type) if args.simi_type !=None else 1
    kmeans_value = int(args.kmeans) if args.kmeans !=None else 5
    run_cluster_resample = int(args.run_cluster_resample) if args.run_cluster_resample !=None else 1
    cluster_type = int(args.cluster_type) if args.cluster_type !=None else 1
    select_feature = float(args.select_feature) if args.select_feature !=None else 1
    # select_feature = float(args.select_feature) if args.select_feature !=None else 1
    x, y, n_feature_2 = load_data(input_file,disease)
    multi_channel(x, y, n_feature_2,cvfold,disease,simi_type,kmeans_value,run_cluster_resample,damping,max_iter,convergence_iter,select_feature,cluster_type)



if __name__=="__main__":

   sys.exit(main())
    
    





       







