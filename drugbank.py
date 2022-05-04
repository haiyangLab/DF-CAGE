import math
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, stats
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve,roc_curve
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
seed = 1
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
# tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)


#drugbank的集子
df = pd.read_csv(r'D:\PycharmProjects\7/gene_list.csv',sep=',')
lis = df['gene'].values.tolist()  #drugbank可用药基因
df1 = pd.read_csv(r'D:\PycharmProjects\CIFAR10\hs_2.csv',sep=',')
df1 =df1[df1['Hugo_Symbol'].isin(lis)]
df1['class'] = 1
df1.to_csv(r'D:\PycharmProjects\7/drugbank_zhen.csv',index=False)
l1 = df1['Hugo_Symbol'].values.tolist()
print(len(df1.index))
df_3 = pd.read_csv(r'D:\PycharmProjects\hm/ffu.csv',sep=',')
print(len(df_3.index))
df_3 = df_3[~df_3['Hugo_Symbol'].isin(l1)]
print(len(df_3.index))
df_3['class'] = 0
df_3 = df_3.sample(n=len(df1.index)*1,random_state=1,replace = False)
df = pd.concat([df1,df_3],axis=0)
df = shuffle(df)
df.to_csv(r'D:\PycharmProjects\7/drugbank_zhenfu.csv',index=False)#,正负样本集

#数据提取
def gaibian (t):
    #可用药的正样本的基因
    df = pd.read_csv(r'D:\PycharmProjects\cna/zhen.csv',sep=',')
    l_zhen = df['Hugo_Symbol'].values.tolist()

    #可疑，负样本的基因
    df_1 = pd.read_csv(r'D:\PycharmProjects\cna/fu.csv',sep=',')
    l_fu = df['Hugo_Symbol'].values.tolist()

    df_2 = pd.read_csv(r'D:\PycharmProjects\hm/ffu.csv',sep=',')
    df_2['class'] = 0
    #将TTN、CACNA1E、COL11A1、DST基因从ffu中排除，这类基因不太可能促进癌症的发生，将其放在负样本中
    l_7= ['TTN','CACNA1E','COL11A1','DST']
    df_7 = df_2[df_2['Hugo_Symbol'].isin(l_7)]
    df_2= df_2[~df_2['Hugo_Symbol'].isin(l_7)]
    df_3 = df_2.sample(n=len(df.index)*t-4,random_state=1,replace = False)
    df = pd.concat([df,df_3],axis=0)  #df为x
    df = pd.concat([df,df_7],axis=0)
    df.to_csv(r'D:\PycharmProjects\hm/zf_ban.csv',index=False,sep=',')

    df_4 = df_2.sample(n=len(df_1.index)*t,random_state=1,replace = False)
    df_1 = pd.concat([df_1,df_4],axis=0)
    df_1['class'] = -1
    df_1.to_csv(r'D:\PycharmProjects\hm/zf_ban_semi.csv',index=False,sep=',')#df_1为semi




df = pd.read_csv(r'D:\PycharmProjects\cna/zonghe_zhenfu.csv',sep=',')
y = df['class'].values
x = df.drop(['biaoqian','Hugo_Symbol','class'], axis = 1).values
df = pd.read_csv(r'D:\PycharmProjects\hm/zf_ban_semi.csv',sep=',')
x_semi = df.drop(['biaoqian','Hugo_Symbol','class'], axis = 1).values



#找roc的最优阈值
def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point

def fisher_ex(a, b, c, d):
    _, pvalue = fisher_exact([[a, b], [c, d]], 'greater')
    if pvalue < 1e-256:
        pvalue = 1e-256
    p1 = -math.log10(pvalue)
    # if pvalue == 0:
    #     p1 = 0
    return p1

#半监督的交叉验证
def fit_cv(X, X_semi, y, k, b_plot=False, method='RF'):
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
        y_semi = np.ones((X_semi.shape[0])) * -1
        X_train = X[~ix, :]
        X_test = X[ix, :]
        # scaler = RobustScaler()
        # X_train = scaler.fit_transform(X_train)
        # X_test = scaler.transform(X_test)



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
            model = SelfTrainingClassifier(CascadeForestClassifier(random_state=1))
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
    # plt.plot(fpr, tpr, label='(our=%0.4f)' % mean_auc)
    return m

#其他方法得auroc
def gui(path,m):
    df = pd.read_csv(r'D:\PycharmProjects\Fashion_MNIST\{}\{}/PANCAN.txt'.format(path,path),sep='\t')
    #if path!= '2020plus':

    for index, row in df.iterrows():
        if row['pvalue'] < 1e-16:
            df.loc[index, 'pvalue'] = 1e-16

    df['pvalue'] = df['pvalue'].apply(np.log10)

    df['pvalue'] = df['pvalue'].apply(lambda x: -x)
    #print(df.head(10))
    # df = df.loc[df['info']=='TYPE=driver'].copy()
    #df = df.loc[df['info'] == 'TYPE=driver']

    df.to_csv(r'D:\PycharmProjects\Fashion_MNIST\{}\{}/PANCAN_1.csv'.format(path, path), index=False, sep=',')


    #//////////////////////////////归一化
    df_guiyi = pd.read_csv(r'D:\PycharmProjects\Fashion_MNIST\{}\{}/PANCAN_1.csv'.format(path,path),sep=',')
    df_guiyi['pvalue']= df_guiyi[['pvalue']].apply(lambda x:(x-np.min(x))/(np.max(x)-np.min(x)))
    #list_1 = df_guiyi['gene'].values.tolist()

    df_new = pd.read_csv(r'D:\PycharmProjects\7/new_zhenfu.csv',sep=',')
    list_new = df_new['Hugo_Symbol'].values.tolist()
    df_guiyi = df_guiyi[df_guiyi['gene'].isin(list_new)]
    df_guiyi = df_guiyi[['gene','pvalue']]
    df_guiyi.rename(columns={'gene':'Hugo_Symbol'},inplace=True)
    df_new = df_new[['Hugo_Symbol','class']]
    df_new = pd.merge(df_new,df_guiyi,how='left',on='Hugo_Symbol')
    df_new['pvalue'] = df_new['pvalue'].fillna(0)
    df_new.to_csv(r'D:\PycharmProjects\Fashion_MNIST\{}\{}/PANCAN_3.csv'.format(path, path), index=False, sep=',')

    #Mann-Whitney U test显著性差异校验
    list_x = df_new['pvalue'].values
    list_y = df_new['class'].values
    x3_all = list_x[list_y == 0]
    x4_all = list_x[list_y == 1]
    statistic, pvalue = stats.mannwhitneyu(x4_all, x3_all, use_continuity=True, alternative='two-sided')

    #ROC指标
    fpr, tpr, thresholds = roc_curve(list_y, list_x)
    roc_auc = auc(fpr, tpr)
    print('{}_pvalue:'.format(path), -math.log10(pvalue))
    print('{}:'.format(path),roc_auc)

    # 富集分析
    df_new_num = df_new[df_new['class'] == 1]['Hugo_Symbol'].tolist()
    df_new1 = df_new[df_new['pvalue'] > m]
    num_a = len(df_new1[df_new1['Hugo_Symbol'].isin(df_new_num)].index)
    num_b = len(df_new1.index) - num_a
    df_num2 = df_new[df_new['pvalue'] < m]
    num_c = len(df_num2[df_num2['Hugo_Symbol'].isin(df_new_num)].index)
    num_d = len(df_new.index) - len(df_new1.index) - num_c
    p = fisher_ex(num_a, num_b, num_c, num_d)
    print('{}富集p：{}'.format(path, p))
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
    m = fit_cv(x, x_semi, y, k=t, method='Deep_RF')
    return m
m = t(1,5)

#加载最优模型对TARGET数据集预测
model = joblib.load('./model.pkl')
#读入DrugbankB数据集
df = pd.read_csv(r'D:\PycharmProjects\7/drugbank_zhenfu.csv',sep=',')#正负样本比例是1：10
l1 = df[df['class']==1]['Hugo_Symbol'].values.tolist()
print(len(l1))
name = df['Hugo_Symbol'].values.tolist()
yy = df['class'].values
xx = df.drop(['biaoqian','Hugo_Symbol','class'], axis = 1).values
#对数据归一化
# scaler =RobustScaler()
# xx = scaler.fit_transform(xx)
#预测
probas_ = model.predict_proba(xx)[:, 1]
dict_from_list = dict(zip(name,probas_))
new = pd.DataFrame(list(dict_from_list.items()))
new.rename(columns={0: 'Hugo_Symbol',
                    1: 'score'}, inplace=True)
new = new[new['Hugo_Symbol'].isin(l1)]
new.sort_values(by='score', ascending=False, inplace=True )
new = new[new['score']>=0.04]
new.to_csv(r'D:\PycharmProjects\7/new.csv',index=False)

#筛选预测集
# l2 = new['Hugo_Symbol'].values.tolist()
l2 = ['JAK1', 'MAPK1', 'JAK2', 'ERBB2', 'FGFR1', 'ALK', 'MAP2K1', 'FGFR3', 'EPHA2', 'KIT', 'PDGFRA', 'ESR1', 'RET', 'CDK4', 'RAF1', 'FGFR2', 'RXRA', 'ERBB4', 'EGFR', 'MET', 'NTRK1', 'BRAF', 'AR', 'PDGFRB', 'ABL1', 'TLR4', 'KDR', 'FLT3', 'CDK6', 'MAP2K2', 'DDR2', 'MMP16', 'TOP1', 'HDAC9', 'MAPT', 'PARP1', 'HDAC1', 'FLT4', 'ATIC', 'MAP2', 'TUBB', 'FLT1', 'TUBA4A', 'TUBA1A', 'MAP4', 'PSMB1', 'TUBG1', 'ESR2', 'TOP2A', 'GART', 'TEK', 'RXRG', 'BTK', 'DPYD', 'SSTR2', 'SSTR1', 'MMP2', 'TUBB1', 'CSF1R', 'PSMB5', 'FCGR2A', 'NR1I2', 'LHCGR', 'VEGFA', 'CD274', 'MMP3', 'CD22', 'ADRA2C', 'RXRB', 'MMP21', 'CSF3R', 'GGPS1', 'FCGR3A', 'TUBD1', 'MMP28', 'FGFR4', 'PDCD1', 'CD80', 'ABAT', 'FRK', 'CD19', 'CD3E', 'CYP11A1', 'DHFR', 'FCGR1A', 'FCGR3B', 'GNRHR', 'MAPK11', 'MRC1', 'NOD2', 'PARP2', 'PARP3', 'PGF', 'RARG', 'RRM1', 'SSTR5', 'TYMP', 'VEGFB']
df1 = pd.read_csv(r'D:\PycharmProjects\CIFAR10\hs_2.csv',sep=',')
df1 =df1[df1['Hugo_Symbol'].isin(l2)]
df1['class'] = 1
df1.to_csv(r'D:\PycharmProjects\7/new_zhen.csv',index=False)
l1 = df1['Hugo_Symbol'].values.tolist()
df_3 = pd.read_csv(r'D:\PycharmProjects\hm/ffu.csv',sep=',')
df_3 = df_3[~df_3['Hugo_Symbol'].isin(l1)]
df_3['class'] = 0
df_3 = df_3.sample(n=len(df1.index)*1,random_state=1,replace = False)
df = pd.concat([df1,df_3],axis=0)
df = shuffle(df)
df.to_csv(r'D:\PycharmProjects\7/new_zhenfu.csv',index=False)#,正负样本集

#预测
y1 = df['class'].values
xx = df.drop(['biaoqian','Hugo_Symbol','class'], axis = 1).values
probas_1 = model.predict_proba(xx)[:, 1]

x1_all = []
x2_all = []
for i in probas_1[y1 == 0]:
    x1_all.append(i)
for i in probas_1[y1 == 1]:
    x2_all.append(i)
statistic, pvalue = stats.mannwhitneyu(x1_all, x2_all, use_continuity=True, alternative='two-sided')
print('DF-CAGE 显著醒检验pvalues ：{}'.format(-math.log10(pvalue)))

#富集分析
probas_array = np.array(probas_1)
probas_df = pd.DataFrame(probas_array)
probas_df.columns = ['pvalues']
df.index = range(len(df))
df1 = pd.concat([df,probas_df],axis=1)
df1_true_num = df1[df1['class']==1]['Hugo_Symbol'].values.tolist()
df2 = df1[df1['pvalues']>m]
num_above_threshold_ture = len(df2[df2['Hugo_Symbol'].isin(df1_true_num)].index)##29(1:1)
num_abave_threshold_false = len(df2.index)-num_above_threshold_ture
df3 = df1[df1['pvalues']<m]
num_below_threshold_ture = len(df3[df3['Hugo_Symbol'].isin(df1_true_num)].index)
num_total = len(df1.index)-len(df2.index)-num_below_threshold_ture
p = fisher_ex(num_above_threshold_ture,num_abave_threshold_false,num_below_threshold_ture,num_total)
print('DF-CAGE富集p：{}'.format(p))

# print(probas_)
fpr, tpr, thresholds = roc_curve(y1, probas_1)
roc_auc_2 = auc(fpr, tpr)
print(roc_auc_2)
plt.plot(fpr, tpr, label='(DF-CAGE=%0.4f)' % roc_auc_2)

file_name = ['OncodriveCLUST','OncodriveFML','MutSig2CV','MuSiC','e-Driver']
for i in file_name:
    gui(i,m)
plt.legend()
plt.show()

df = pd.read_csv(r'D:\PycharmProjects\7/drugbank_zhen.csv',sep=',')
df = df[~df['Hugo_Symbol'].isin(l2)]
print(df['Hugo_Symbol'].tolist());print(len(df['Hugo_Symbol'].tolist()))
print(l2);print(len(l2))