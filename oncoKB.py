import joblib
import numpy as np
import pandas as pd
from scipy.stats import stats, fisher_exact,mannwhitneyu
from sklearn import preprocessing
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve,roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from deepforest import CascadeForestClassifier
from sklearn.neural_network import MLPClassifier
import math

#Data Integration
def gaibian (t):
    # Gene of drug-available positive samples (oncokb)
    df = pd.read_csv(r'.\input\zhen.csv',sep=',')
    l_zhen = df['Hugo_Symbol'].values.tolist()
    df_2 = pd.read_csv(r'.\input\ffu.csv',sep=',')
    df_2['class'] = 0

    # Exclude TTN, CACNA1E, COL11A1, DST genes from ffu, which are unlikely to contribute to cancer,
    # and put them in negative samples
    l_7= ['TTN','CACNA1E','COL11A1','DST']
    df_7 = df_2[df_2['Hugo_Symbol'].isin(l_7)]
    df_2= df_2[~df_2['Hugo_Symbol'].isin(l_7)]
    df_3 = df_2.sample(n=len(df.index)*t-4,random_state=1,replace = False)
    df = pd.concat([df,df_3],axis=0)  #df为x
    df = pd.concat([df,df_7],axis=0)
    df.to_csv(r'.\results\zf_ban.csv',index=False,sep=',')









def get_median(data):
   data.sort()
   half = len(data) // 2
   return (data[half] + data[~half]) / 2

#Find the optimal threshold of roc
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
    # p1 = -math.log10(pvalue)
    # if pvalue == 0:
    #     p1 = 0
    return pvalue

#Cross-validation
def fit_cv(X,y, k,a,b,c,d, b_plot=False, method='RF'):
    n = X.shape[0]
    assignments = np.array((n // k + 1) * list(range(1, k + 1)))
    assignments = assignments[:n]
    mean_tpr = 0.0
    tprs = []
    p = []
    x1_all = []
    x2_all = []
    mean_fpr = np.linspace(0, 1, 100)
    z = 0
    roc_auc = 0
    for i in range(1, k + 1):
        ix = assignments == i
        y_test = y[ix]
        y_train = y[~ix]
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
        df = pd.read_csv(r'.\results\zf_ban.csv',sep=',')
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

            # record optimal threshold
            roc_auc_1 = auc(fpr, tpr)
            print('AUROC：{}'.format(roc_auc_1))
            optimal_th, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)
            print('threshold：{}'.format(optimal_th))
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

        #enrichment analysis
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

    # enrichment analysis
    p = fisher_ex(a, b, c, d)
    print('DF-CAG:Eenrichment_pvalue：{}'.format(p))

    # print(mean_pvalue)
    statistic, pvalue = mannwhitneyu(x1_all, x2_all, use_continuity=True, alternative='two-sided')
    mean_auc = auc(mean_fpr, mean_tpr)
    print('roc_pvalue: {}'.format(pvalue))
    print("DF-CAGE Mean AUPRC (area = %0.4f)" % (mean_auc))
    plt.plot(fpr, tpr, label='(DF-CAGE=%0.4f)' % mean_auc,)
    return m

#其他方法得auroc
def gui(path,m):
    df = pd.read_csv(r'.\model\{}\{}/PANCAN.txt'.format(path,path),sep='\t')
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

    df.to_csv(r'.\model\{}\{}/PANCAN_1.csv'.format(path, path), index=False, sep=',')


    #//////////////////////////////归一化
    df_guiyi = pd.read_csv(r'.\model\{}\{}/PANCAN_1.csv'.format(path,path),sep=',')
    #进行从大到小排序，然后把重复项去除，保留数值大的
    df_guiyi.sort_values(by='pvalue', ascending=False,inplace=True)
    df_guiyi = df_guiyi.drop_duplicates(subset=['gene'], keep='first')
    df_guiyi['pvalue']= df_guiyi[['pvalue']].apply(lambda x:(x-np.min(x))/(np.max(x)-np.min(x)))
    #list_1 = df_guiyi['gene'].values.tolist()

    df_new = pd.read_csv(r'.\results\zf_ban.csv',sep=',')
    list_new = df_new['Hugo_Symbol'].values.tolist()
    df_guiyi = df_guiyi[df_guiyi['gene'].isin(list_new)]
    df_guiyi = df_guiyi[['gene','pvalue']]
    df_guiyi.rename(columns={'gene':'Hugo_Symbol'},inplace=True)
    df_new = df_new[['Hugo_Symbol','class']]
    df_new = pd.merge(df_new,df_guiyi,how='left',on='Hugo_Symbol')
    df_new['pvalue'] = df_new['pvalue'].fillna(df_new['pvalue'].mean())
    # df_new['pvalue'] = df_new['pvalue'].fillna(0)
    df_new.to_csv(r'.\model\{}\{}/PANCAN_3.csv'.format(path, path), index=False, sep=',')

    list_x = df_new['pvalue'].values
    list_y = df_new['class'].values
    x3_all = list_x[list_y == 0]
    x4_all = list_x[list_y == 1]
    statistic, pvalue = mannwhitneyu(x4_all, x3_all, use_continuity=True, alternative='two-sided')
    fpr, tpr, thresholds = roc_curve(list_y, list_x)
    roc_auc = auc(fpr, tpr)

    #enrichment analysis
    df_new_num = df_new[df_new['class'] == 1]['Hugo_Symbol'].tolist()
    df_new1 = df_new[df_new['pvalue'] > m]
    num_a = len(df_new1[df_new1['Hugo_Symbol'].isin(df_new_num)].index)
    num_b = len(df_new1.index) - num_a
    df_num2 = df_new[df_new['pvalue'] < m]
    num_c = len(df_num2[df_num2['Hugo_Symbol'].isin(df_new_num)].index)
    num_d = len(df_new.index) - len(df_new1.index) - num_c
    p = fisher_ex(num_a, num_b, num_c, num_d)
    print('{}_enrichment_pvalue：{}'.format(path, p))
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
    df = pd.read_csv(r'.\results\zf_ban.csv', sep=',')
    y = df['class'].values
    x = df.drop(['biaoqian', 'Hugo_Symbol', 'class'], axis=1).values
    m = fit_cv(x,y,t,0,0,0,0, method='Deep_RF')
    return m
m = t(1,5)
file_name = ['OncodriveCLUST','OncodriveFML','MutSig2CV','MuSiC','e-Driver']
for i in file_name:
    gui(i,m)
plt.grid(False)
plt.legend()
plt.show()