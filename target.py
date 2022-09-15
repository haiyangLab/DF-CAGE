import numpy as np
import pandas as pd
from scipy.stats import stats
from sklearn.preprocessing import RobustScaler,MinMaxScaler
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve,roc_curve,average_precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from deepforest import CascadeForestClassifier
from scipy.stats import fisher_exact
import joblib
import os
import random
import torch
import math
#设置随机种子
seed = 1
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
# tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

def gaibian (t):
    df = pd.read_csv(r'./input/positive.csv',sep=',')
    l_zhen = df['Hugo_Symbol'].values.tolist()

    df_2 = pd.read_csv(r'./input/negative.csv',sep=',')
    df_2['class'] = 0
    l_7= ['TTN','CACNA1E','COL11A1','DST']
    df_7 = df_2[df_2['Hugo_Symbol'].isin(l_7)]
    df_2= df_2[~df_2['Hugo_Symbol'].isin(l_7)]
    df_3 = df_2.sample(n=len(df.index)*t-4,random_state=1,replace = False)
    df = pd.concat([df,df_3],axis=0)  #df为x
    df = pd.concat([df,df_7],axis=0)
    df.to_csv(r'./results/oncokb_set.csv',index=False,sep=',')




df = pd.read_csv(r'./results/oncokb_set.csv',sep=',')
y = df['class'].values
x = df.drop(['label','Hugo_Symbol','class'], axis = 1).values

#找roc的最优阈值
def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point

def fisher_ex(a, b, c, d):
    ordio, pvalue = fisher_exact([[a, b], [c, d]], 'greater')
    if pvalue < 1e-256:
        pvalue = 1e-256
    # p1 = -math.log10(pvalue)
    return ordio,pvalue

#p存储模型的预测值，q存储标签
p = []
q = []

#半监督的交叉验证
def fit_cv(X,y, k, b_plot=False, method='RF'):
    n = X.shape[0]
    assignments = np.array((n // k + 1) * list(range(1, k + 1)))
    assignments = assignments[:n]
    tprs = []
    z = 0
    mean_fpr = np.linspace(0, 1, 100)
    for i in range(1, k + 1):
        ix = assignments == i
        y_test = y[ix]
        y_train = y[~ix]
        X_train = X[~ix, :]
        X_test = X[ix, :]
        scaler = RobustScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)



        # sys.exit()

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
            # model = SelfTrainingClassifier(CascadeForestClassifier(random_state=1))
            model = CascadeForestClassifier(random_state=1)

            model.fit(X_train, y_train)
            probas_ = model.predict_proba(X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, probas_)
            roc_auc_1 = auc(fpr, tpr)
            print('准确率：{}'.format(roc_auc_1))
            optimal_th, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)
            print('门限：{}'.format(optimal_th))
            if roc_auc_1 > z:
                z = roc_auc_1
                m = optimal_th
                joblib.dump(model, './model.pkl')
            # print(probas_)
            del model
        y_test = y_test.tolist()
        p.extend(probas_)
        q.extend(y_test)

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
    print("Mean AUroC (area = %0.4f)" % mean_auc)
    plt.plot(fpr, tpr, label='(our=%0.4f)' % mean_auc)
    precision, recall, thresholds = precision_recall_curve(q, p)
    AP = average_precision_score(y_test, probas_)
    print("Mean AUPRC (area = %0.4f)" % AP)
    return m

#其他方法得auroc
def gui(path,m):
    df = pd.read_csv(r'./models/{}/{}/PANCAN.txt'.format(path,path),sep='\t')
    if path == '2020plus':
        df = df.loc[df['info'] == 'TYPE=oncogene']

    for index, row in df.iterrows():
        if row['pvalue'] < 1e-16:
            df.loc[index, 'pvalue'] = 1e-16

    df['pvalue'] = df['pvalue'].apply(np.log10)

    df['pvalue'] = df['pvalue'].apply(lambda x: -x)
    #print(df.head(10))
    # df = df.loc[df['info']=='TYPE=driver'].copy()
    #df = df.loc[df['info'] == 'TYPE=driver']

    df.to_csv(r'./models/{}/{}/PANCAN_1.csv'.format(path, path), index=False, sep=',')


    #//////////////////////////////归一化
    df_guiyi = pd.read_csv(r'./models/{}/{}/PANCAN_1.csv'.format(path,path),sep=',')
    #进行从大到小排序，然后把重复项去除，保留数值大的
    df_guiyi.sort_values(by='pvalue', ascending=False,inplace=True)
    df_guiyi = df_guiyi.drop_duplicates(subset=['gene'], keep='first')
    df_guiyi['pvalue']= df_guiyi[['pvalue']].apply(lambda x:(x-np.min(x))/(np.max(x)-np.min(x)))
    #list_1 = df_guiyi['gene'].values.tolist()

    df_new = pd.read_csv(r'./input/target_set.csv', sep=',')
    print(len(df_new.index))
    list_new = df_new['Hugo_Symbol'].values.tolist()
    df_guiyi = df_guiyi[df_guiyi['gene'].isin(list_new)]
    df_guiyi = df_guiyi[['gene','pvalue']]
    df_guiyi.rename(columns={'gene':'Hugo_Symbol'},inplace=True)
    df_new = df_new[['Hugo_Symbol','class']]
    df_new = pd.merge(df_new,df_guiyi,how='left',on='Hugo_Symbol')
    df_new['pvalue'] = df_new['pvalue'].fillna(df_new['pvalue'].mean())
    # df_new['pvalue'] = df_new['pvalue'].fillna(0)
    df_new.to_csv(r'D./models/{}/{}/PANCAN_3.csv'.format(path, path), index=False, sep=',')
    print(len(df_new.index))
    list_x = df_new['pvalue'].values
    list_y = df_new['class'].values
    x3_all = list_x[list_y == 0]
    x4_all = list_x[list_y == 1]
    statistic, pvalue = stats.mannwhitneyu(x4_all, x3_all, use_continuity=True, alternative='two-sided')
    fpr, tpr, thresholds = roc_curve(list_y, list_x)
    # roc_auc = auc(fpr, tpr)
    # print('{}_pvalue:'.format(path),pvalue)
    # print('{}:'.format(path),roc_auc)

    #AUPRC
    precision, recall, thresholds = precision_recall_curve(list_y, list_x)
    AP = average_precision_score(list_y, list_x)
    print('AUPRC--{}:'.format(path), AP)

    #富集分析
    df_new_num = df_new[df_new['class']==1]['Hugo_Symbol'].tolist()
    df_new1 = df_new[df_new['pvalue']>m]
    num_a = len(df_new1[df_new1['Hugo_Symbol'].isin(df_new_num)].index)
    print('{}方法报告的可用药基因个数：{}'.format(path, num_a))
    num_b = len(df_new1.index)-num_a
    df_num2 = df_new[df_new['pvalue']<m]
    num_c = len(df_num2[df_num2['Hugo_Symbol'].isin(df_new_num)].index)
    num_d = len(df_new.index)-len(df_new1.index)-num_c
    o,p = fisher_ex(num_a,num_b,num_c,num_d)
    print('{}富集p：{}'.format(path,p))
    # odds ratio
    print('{}富集odds ratio：{}'.format(path, o))

    #AP = average_precision_score(list_y, list_x)
    #plt.plot(recall, precision, label='{}(AP=%0.2f)'.format(path) % AP)
    plt.plot(fpr, tpr, label='{}(AUC=%0.4f)'.format(path) % roc_auc)



def t(k,t):
    gaibian(k)
    plt.figure()
    plt.grid()
    plt.title('1:{}  {}cross   Roc Curve'.format(k,t))
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    m = fit_cv(x, y, k=t, method='Deep_RF')
    return m
m = t(20,5)
print(m)
#Load the optimal model to predict the TARGET dataset
model = joblib.load('./model.pkl')

#Read in the TRAGET dataset
df = pd.read_csv(r'./input/target_set.csv',sep=',')
yy = df['class'].values
xx = df.drop(['biaoqian','Hugo_Symbol','class'], axis = 1).values

#normalize the data
scaler =RobustScaler()
xx = scaler.fit_transform(xx)

#predict
probas_ = model.predict_proba(xx)[:, 1]
x1_all = []
x2_all = []
for i in probas_[yy == 0]:
    x1_all.append(i)
for i in probas_[yy == 1]:
    x2_all.append(i)
statistic, pvalue = stats.mannwhitneyu(x1_all, x2_all, use_continuity=True, alternative='two-sided')

#Fisher's enrichment analysis
probas_array = np.array(probas_)
probas_df = pd.DataFrame(probas_array)
probas_df.columns = ['pvalues']
df1 = pd.concat([df,probas_df],axis=1)

df1_true_num = df1[df1['class']==1]['Hugo_Symbol'].values.tolist()
df2 = df1[df1['pvalues']>m]
#print('DF-CAGE方法报告的可用药基因的个数：{}'.format(len(df2.index)))
num_above_threshold_ture = len(df2[df2['Hugo_Symbol'].isin(df1_true_num)].index)##29(1:1)
print('DF-CAGE方法报告的可用药基因的个数：{}'.format(num_above_threshold_ture))
num_abave_threshold_false = len(df2.index)-num_above_threshold_ture
df3 = df1[df1['pvalues']<m]
num_below_threshold_ture = len(df3[df3['Hugo_Symbol'].isin(df1_true_num)].index)
num_total = len(df1.index)-len(df2.index)-num_below_threshold_ture
o,p = fisher_ex(num_above_threshold_ture,num_abave_threshold_false,num_below_threshold_ture,num_total)
print('DF-CAGE的富集p：{}'.format(p))
print('DF-CAGE的富集odds ratio：{}'.format(o))



fpr, tpr, thresholds = roc_curve(yy, probas_)
roc_auc_2 = auc(fpr, tpr)
#print('DF-CAGE方法的pvalue: {}'.format(-math.log10(pvalue)))
print('DF-CAGE方法的pvalue: {}'.format(pvalue))
print('DF-CAGE方法的roc_auc_2: {}'.format(roc_auc_2))
precision, recall, thresholds = precision_recall_curve(yy, probas_)
AP = average_precision_score(yy, probas_)
print("DF-CAGE方法的AUPRC (area = %0.4f)" % AP)

file_name = ['OncodriveCLUST','OncodriveFML','MutSig2CV','MuSiC','e-Driver']
for i in file_name:
    gui(i,m)

#DriverML
df = pd.read_csv(r'./input/target_set.csv',sep=',')
l0 = df['Hugo_Symbol'].values.tolist()
print(len(l0))
df1 = pd.read_csv(r'D:\PycharmProjects\补充实验\DriverML.csv',sep=',')

df1 = df1[df1['gene'].isin(l0)]
df1.rename(columns={'gene':'Hugo_Symbol'},inplace=True)
df_new = pd.merge(df,df1,how='left',on='Hugo_Symbol')
df_new = df_new[['score','class']]
df_new['score'] = df_new['score'].fillna(df_new['score'].mean())
list_x = df_new['score'].values
list_y = df_new['class'].values
#AUROC
fpr, tpr, thresholds = roc_curve(list_y, list_x)
roc_auc = auc(fpr, tpr)
print('DriverML的AUROC: {}'.format(roc_auc))
plt.plot(fpr, tpr, label='DriverML (AUC=%0.4f)' % roc_auc)

#AUPRC
precision, recall, thresholds = precision_recall_curve(list_y, list_x)
AP = average_precision_score(list_y, list_x)
print("DriverML的AUPRC (area = %0.4f)" % AP)
#plt.plot(recall, precision, label='DriverML (AUPRC=%0.2f)' % AP)

plt.legend()
plt.show()