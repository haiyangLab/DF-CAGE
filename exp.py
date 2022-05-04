import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from deepforest import CascadeForestClassifier
import joblib
import os
import random
import torch
from sklearn.utils import shuffle

#设置随机种子
seed = 6
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
# tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

#oncoKB和target的并集,作为可用药的正样本
df = pd.read_csv(r'.\input\new_set_zhen.csv',sep=',')#target的可用药基因
ll = df['Hugo_Symbol'].values.tolist()
df1 = pd.read_csv(r'.\input\zhen.csv',sep=',')
ll1 = df1['Hugo_Symbol'].values.tolist()
df = df[~df['Hugo_Symbol'].isin(ll1)]
print(len(df['Hugo_Symbol'].tolist()))


df = pd.concat([df,df1],axis=0)
df = shuffle(df)
df.to_csv(r'.\results\zong_zhenfu.csv',index=False,)#,oncoKB和target的并集

df_3 = pd.read_csv(r'.\input\ffu.csv',sep=',')
df_3['class'] = 0
#将TTN、CACNA1E、COL11A1、DST基因从ffu中排除，这类基因不太可能促进癌症的发生，将其放在负样本中
l_7= ['TTN','CACNA1E','COL11A1','DST']
df_7 = df_3[df_3['Hugo_Symbol'].isin(l_7)]
df_3= df_3[~df_3['Hugo_Symbol'].isin(l_7)]
df_3 = df_3.sample(n=len(df.index)*10-4,random_state=1,replace = False)
df = pd.concat([df,df_3],axis=0)
df = pd.concat([df,df_7],axis=0)
df = shuffle(df)
df.to_csv(r'.\results\zonghe_zhenfu_1.csv',index=False)#,正负样本集

# Find the optimal threshold of roc
def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point

# Cross validation
def fit_cv(X, y, k, b_plot=False, method='RF'):

    n = X.shape[0]
    assignments = np.array((n // k + 1) * list(range(1, k + 1)))
    print(assignments)
    assignments = assignments[:n]
    mean_tpr = 0.0
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    roc_auc = 0
    z = 0
    men = 0
    for i in range(1, k + 1):
        ix = assignments == i
        y_test = y[ix]
        print(y_test[:100])
        y_train = y[~ix]
        print(len(y_train))
        print(y_train)
        X_train = X[~ix, :]
        X_test = X[ix, :]
        # scaler = preprocessing.MinMaxScaler()
        # X_train = scaler.fit_transform(X_train)
        # X_test = scaler.transform(X_test)

        if method == 'SVM':
            model = SelfTrainingClassifier(SVC(gamma='auto', probability=True))
            model.fit(X_train, y_train)
            probas_ = model.predict_proba(X_test)[:, 1]
            del model
        elif method == 'RF':
            model = SelfTrainingClassifier(
                RandomForestClassifier( random_state=0,))
            model.fit(X_train, y_train)
            probas_ = model.predict_proba(X_test)[:, 1]
            del model
        elif method == 'Deep_RF':
            model = CascadeForestClassifier(random_state=1)
            model.fit(X_train, y_train)
            probas_ = model.predict_proba(X_test)[:, 1]

            fpr, tpr, thresholds = roc_curve(y_test, probas_)
            roc_auc_1 = auc(fpr, tpr)
            print('AUROC ：{}' .format(roc_auc_1))
            optimal_th, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)

            print('threshold ：{}'.format(optimal_th))
            if roc_auc_1 > z:
                z = roc_auc_1
                m =optimal_th
                joblib.dump(model, './model_1.pkl')

            del model

        fpr, tpr, thresholds = roc_curve(y_test, probas_)
        #prs.append(np.interp(mean_recall, fpr, tpr))
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        #roc_auc += auc(fpr, tpr)
        #aucs.append(roc_auc)

    #mean_precision = np.mean(prs, axis=0)
    #mean_auc = auc(mean_recall, mean_precision)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    # mean_auc = roc_auc/k
    print("Mean AUPRC (area = %0.4f)" % mean_auc)
    plt.plot(fpr, tpr, label='(our=%0.4f)' % mean_auc)
    plt.show()
    return mean_auc,m

# Get training data
df = pd.read_csv(r'.\results\zonghe_zhenfu_1.csv',sep=',')
y = df['class'].values
x = df.drop(['biaoqian','Hugo_Symbol','class'], axis = 1).values

# The model is trained and thresholds are obtained
m ,n = fit_cv(x, y, k=5, method='Deep_RF')
# m ,n = fit_cv(x, y, k=5, method='SVM')


# Score ~ 20,000 genes
df = pd.read_csv(r'.\input\hs_2.csv', sep=',')
xsemi = df.drop(['biaoqian', 'Hugo_Symbol'], axis=1).values

# Load the optimal model to predict the TARGET dataset
model = joblib.load('./model_1.pkl')
probas = model.predict_proba(xsemi)[:, 1]
print(n)
list1 = np.where(probas >= n)[0].tolist()
print(len(list1))
df1 = df.iloc[list1, :]
list_1 = df1['Hugo_Symbol'].values.tolist()
print(list_1)
print(len(list_1))

