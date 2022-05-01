import joblib
import numpy as np
import pandas as pd
from scipy.stats import stats, fisher_exact
from sklearn import preprocessing
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve,roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from deepforest import CascadeForestClassifier
from sklearn.neural_network import MLPClassifier
import math

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




df = pd.read_csv(r'D:\PycharmProjects\hm/zf_ban.csv',sep=',')
y = df['class'].values
x = df.drop(['biaoqian','Hugo_Symbol','class'], axis = 1).values
df = pd.read_csv(r'D:\PycharmProjects\hm/zf_ban_semi.csv',sep=',')
x_semi = df.drop(['biaoqian','Hugo_Symbol','class'], axis = 1).values

#get-median
def get_median(data):
   data.sort()
   half = len(data) // 2
   return (data[half] + data[~half]) / 2

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
def fit_cv(X, X_semi, y, k,a,b,c,d, b_plot=False, method='RF'):
    n = X.shape[0]
    assignments = np.array((n // k + 1) * list(range(1, k + 1)))
    assignments = assignments[:n]
    mean_tpr = 0.0
    # all_tpr = []
    # all_roc = []
    #auprc_list = []
    #prs = []
    #aucs = []
    tprs = []
    p = []
    x1_all = []
    x2_all = []
    mean_fpr = np.linspace(0, 1, 100)
    z = 0
    roc_auc = 0
    #mean_recall = np.linspace(0, 1, 100)
    for i in range(1, k + 1):
        ix = assignments == i
        y_test = y[ix]
        y_train = y[~ix]
        y_semi = np.ones((X_semi.shape[0])) * -1
        # X = X.reshape(X.size)
        X_train = X[~ix, :]
        # print(X_train.shape)
        # print(y_train)
        # X_train = np.concatenate([X_train, X_semi])
        # y_train = np.concatenate([y_train, y_semi])
        # print(y_train)
        X_test = X[ix, :]
        scaler = preprocessing.MinMaxScaler()
        #pca_model = PCA(n_components=64)
        X_train = scaler.fit_transform(X_train)
        # print()
        #X_train = pca_model.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        # print(X_test)
        #X_test = pca_model.transform(X_test)
        xtext = ix.nonzero()[0].tolist()
        df = pd.read_csv(r'D:\PycharmProjects\hm/zf_ban.csv',sep=',')
        df0 = df.iloc[xtext,:]
        # sys.exit()

        if method == 'SVM':
            model = SVC(gamma='auto', probability=True)
            model.fit(X_train, y_train)
            probas_ = model.predict_proba(X_test)[:, 1]
            del model
        elif method =='mlp':
            model = MLPClassifier(hidden_layer_sizes=[2], activation='relu', solver='adam', random_state=0,batch_size=78, max_iter=1000,)
            model.fit(X_train, y_train)
            probas_ = model.predict_proba(X_test)[:, 1]
        elif method == 'RF':
            model = RandomForestClassifier( n_estimators=30,random_state=0)
            model.fit(X_train, y_train)
            probas_ = model.predict_proba(X_test)[:, 1]
            del model
        elif method == 'Deep_RF':
            # model = SelfTrainingClassifier(CascadeForestClassifier(random_state=1))
            model = CascadeForestClassifier(random_state=1)
            model.fit(X_train, y_train)
            probas_ = model.predict_proba(X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, probas_)

            # 记录最优门限
            roc_auc_1 = auc(fpr, tpr)
            print('准确率：{}'.format(roc_auc_1))
            optimal_th, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)
            print('门限：{}'.format(optimal_th))
            if roc_auc_1 > z:
                z = roc_auc_1
                m = optimal_th
                joblib.dump(model, './model_KB.pkl')

            del model
        for i in probas_[y_test==0]:
            x1_all.append(i)
        for i in probas_[y_test==1]:
            x2_all.append(i)

        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)

        #富集分析
        optimal_th, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)
        probas_array = np.array(probas_)
        probas_df = pd.DataFrame(probas_array)
        probas_df.columns = ['pvalues']
        df0.index = range(len(df0))
        df1 = pd.concat([df0, probas_df], axis=1)
        df1_true_num = df1[df1['class'] == 1]['Hugo_Symbol'].values.tolist()
        df2 = df1[df1['pvalues'] > optimal_th]
        num_above_threshold_ture = len(df2[df2['Hugo_Symbol'].isin(df1_true_num)].index)  ##29(1:1)
        a += num_above_threshold_ture
        num_abave_threshold_false = len(df2.index) - num_above_threshold_ture
        b += num_abave_threshold_false
        df3 = df1[df1['pvalues'] < optimal_th]
        num_below_threshold_ture = len(df3[df3['Hugo_Symbol'].isin(df1_true_num)].index)
        c += num_below_threshold_ture
        num_total = len(df1.index) - len(df2.index) - num_below_threshold_ture
        d += num_total

   #mean_precision = np.mean(prs, axis=0)
    #mean_auc = auc(mean_recall, mean_precision)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0

    # 富集分析
    p = fisher_ex(a, b, c, d)
    print('DF-CAGE富集p：{}'.format(p))

    # print(mean_pvalue)
    statistic, pvalue = stats.mannwhitneyu(x1_all, x2_all, use_continuity=True, alternative='two-sided')
    mean_auc = auc(mean_fpr, mean_tpr)
    print('roc_pvalue: {}'.format(pvalue))
    print("Mean AUPRC (area = %0.4f),p-value = %f" % (mean_auc,pvalue))
    plt.plot(fpr, tpr, label='(DF-CAGE=%0.4f)' % mean_auc,)
    return m

#其他方法得auroc
def gui(path,m):
    df = pd.read_csv(r'D:\PycharmProjects\Fashion_MNIST\{}\{}/PANCAN.txt'.format(path,path),sep='\t')
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

    df.to_csv(r'D:\PycharmProjects\Fashion_MNIST\{}\{}/PANCAN_1.csv'.format(path, path), index=False, sep=',')


    #//////////////////////////////归一化
    df_guiyi = pd.read_csv(r'D:\PycharmProjects\Fashion_MNIST\{}\{}/PANCAN_1.csv'.format(path,path),sep=',')
    #进行从大到小排序，然后把重复项去除，保留数值大的
    df_guiyi.sort_values(by='pvalue', ascending=False,inplace=True)
    df_guiyi = df_guiyi.drop_duplicates(subset=['gene'], keep='first')
    df_guiyi['pvalue']= df_guiyi[['pvalue']].apply(lambda x:(x-np.min(x))/(np.max(x)-np.min(x)))
    #list_1 = df_guiyi['gene'].values.tolist()

    df_new = pd.read_csv(r'D:\PycharmProjects\hm/zf_ban.csv',sep=',')
    list_new = df_new['Hugo_Symbol'].values.tolist()
    df_guiyi = df_guiyi[df_guiyi['gene'].isin(list_new)]
    df_guiyi = df_guiyi[['gene','pvalue']]
    df_guiyi.rename(columns={'gene':'Hugo_Symbol'},inplace=True)
    df_new = df_new[['Hugo_Symbol','class']]
    df_new = pd.merge(df_new,df_guiyi,how='left',on='Hugo_Symbol')
    df_new['pvalue'] = df_new['pvalue'].fillna(df_new['pvalue'].mean())
    # df_new['pvalue'] = df_new['pvalue'].fillna(0)
    df_new.to_csv(r'D:\PycharmProjects\Fashion_MNIST\{}\{}/PANCAN_3.csv'.format(path, path), index=False, sep=',')

    list_x = df_new['pvalue'].values
    list_y = df_new['class'].values
    x3_all = list_x[list_y == 0]
    x4_all = list_x[list_y == 1]
    statistic, pvalue = stats.mannwhitneyu(x4_all, x3_all, use_continuity=True, alternative='two-sided')
    fpr, tpr, thresholds = roc_curve(list_y, list_x)
    roc_auc = auc(fpr, tpr)

    #富集分析
    df_new_num = df_new[df_new['class'] == 1]['Hugo_Symbol'].tolist()
    df_new1 = df_new[df_new['pvalue'] > m]
    num_a = len(df_new1[df_new1['Hugo_Symbol'].isin(df_new_num)].index)
    num_b = len(df_new1.index) - num_a
    df_num2 = df_new[df_new['pvalue'] < m]
    num_c = len(df_num2[df_num2['Hugo_Symbol'].isin(df_new_num)].index)
    num_d = len(df_new.index) - len(df_new1.index) - num_c
    p = fisher_ex(num_a, num_b, num_c, num_d)
    print('{}富集p：{}'.format(path, p))
    ###

    print('{}_pvalue:'.format(path),pvalue)
    print('{}:'.format(path),roc_auc)
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
    m = fit_cv(x, x_semi, y,t,0,0,0,0, method='Deep_RF')
    return m
m = t(1,5)
file_name = ['OncodriveCLUST','OncodriveFML','MutSig2CV','MuSiC','e-Driver']
for i in file_name:
    gui(i,m)
plt.grid(False)
plt.legend()
plt.show()