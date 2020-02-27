import pickle
import pandas as pd
import numpy as np
import copy
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from score import report_score
from sklearn.linear_model import LogisticRegression
#_____________________
##from pandas.plotting import scatter_matrix
##import matplotlib.pyplot as plt
##from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
##from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
##from sklearn.preprocessing import normalize
##from sklearn.metrics.pairwise import cosine_similarity
#_____________________

TRAIN_H5 = "prs_trn_2.h5"
F_H5 = "prs_comp_tst.h5"
F_PKL = 'test_feature.pkl'

f = open('feature.pkl', 'rb')
trainX_all = pickle.load(f)
f.close()
combined_dataframe = pd.read_hdf(TRAIN_H5)
#combined_dataframe = pd.read_hdf("prs_test_2.h5")
print(combined_dataframe, combined_dataframe.info())
combined_dataframe['first']=combined_dataframe['Stance'].apply({'unrelated':1,'discuss':0,'agree':0,'disagree':0}.get)
combined_dataframe['second'] = combined_dataframe['Stance'].apply({'unrelated':0,'discuss':1,'agree':2,'disagree':3}.get)
trainy_all = list((combined_dataframe.values[:,4]).astype('int64'))
stage2 = combined_dataframe[combined_dataframe['first']==0].index.tolist()
stage2_frame = combined_dataframe[combined_dataframe['first']==0]
trainX = []
for i in stage2:
    trainX.append(trainX_all[i])
trainY = list((stage2_frame.values[:,5]).astype('int64'))


# trainX_all and trainy_all is for binary classification
# trainX and trainY is for triple classification


#method1: 2 classification for all 4 class
import xgboost as xgb
def train_relatedness_classifier(trainX, trainY):
    xg_train = xgb.DMatrix(trainX, label=trainY)
    # setup parameters for xgboost
    param = {}
    # use softmax multi-class classification
    param['objective'] = 'binary:logistic'
    # scale weight of positive examples
    param['eta'] = 0.1
    param['max_depth'] = 6
    param['silent'] = 1
    param['nthread'] = 20

    num_round = 1000
    relatedness_classifier = xgb.train(param, xg_train, num_round)
    return relatedness_classifier
def method1():
    all_trainX = copy.deepcopy(trainX_all)
    all_trainY = list((combined_dataframe.values[:,4]).astype('int64'))
    relatedness_classifier = train_relatedness_classifier(all_trainX, all_trainY)

    relatedTrainX = trainX
    related_frame = combined_dataframe[combined_dataframe['first']==0]
    related_frame["discussY"] = related_frame['Stance'].apply({'discuss':1,'agree':0,'disagree':0}.get)
    relatedTrainY = list((related_frame.values[:,6]).astype('int64'))
    discuss_classifier = train_relatedness_classifier(relatedTrainX, relatedTrainY)

    agree = combined_dataframe[combined_dataframe['second']>1].index.tolist()
    agreeTrainX = []
    for i in agree:
        agreeTrainX.append(trainX_all[i])
    agree_frame = combined_dataframe[combined_dataframe['second']>1]
    agree_frame["agreeY"] = agree_frame['Stance'].apply({'agree':1,'disagree':0}.get)
    agreeTrainY = (agree_frame.values[:,6]).astype('int64')
    agree_classifier = train_relatedness_classifier(agreeTrainX, agreeTrainY)
    return relatedness_classifier,discuss_classifier,agree_classifier
def prediction1():
    f = open('test_feature.pkl', 'rb')
    textX = pickle.load(f)
    f.close()
    f = open('relatedness_classifier.pkl', 'rb')
    relatedness_classifier = pickle.load(f)
    f.close()
    f = open('discuss_classifier.pkl', 'rb')
    discuss_classifier = pickle.load(f)
    f.close()
    f = open('agree_classifier.pkl', 'rb')
    agree_classifier = pickle.load(f)
    f.close()
    xg_test = xgb.DMatrix(textX)
    relatedness_pred = relatedness_classifier.predict(xg_test);
    discuss_pred = discuss_classifier.predict(xg_test)
    agree_pred = agree_classifier.predict(xg_test)

    ret, scores = [], []
    for (pred_relate, pred_discuss, pred_agree) in zip(relatedness_pred, discuss_pred, agree_pred):
        scores.append((pred_relate, pred_discuss, pred_agree))
        if pred_relate >= 0.5:
            ret.append('unrelated')
        elif pred_discuss >= 0.5:
            ret.append('discuss')
        elif pred_agree >= 0.5:
            ret.append('agree')
        else:
            ret.append('disagree')
    return relatedness_pred,discuss_pred,agree_pred,ret,scores

#method2:  2 classification + 3 classification
def binary():
    lr= LogisticRegression(C = 1.0,penalty = 'l2',solver = 'lbfgs')
    metric_all = pd.DataFrame()
    metric = cross_val_score(lr, trainX_all,trainy_all, cv=10, scoring='f1',verbose = 1)
    metric.sort()
    metric_all['glm'] = metric[::-1]


    X_train1, X_test1, y_train1, y_test1 = train_test_split(trainX_all,trainy_all, test_size = 0.3, random_state = 0)
    svc = SVC(C=1.0, kernel='rbf', gamma='auto')
    metric = cross_val_score(svc, X_test1,y_test1, cv=10, scoring='f1',verbose = 1)
    #[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    #[Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed:  1.8min finished
    metric.sort()
    metric_all['svm'] = metric[::-1]

    dt = DecisionTreeClassifier(criterion="gini",max_depth=11)
    X_train1, X_test1, y_train1, y_test1 = train_test_split(trainX_all,trainy_all, test_size = 0.3, random_state = 0)
    metric = cross_val_score(dt, X_train1,y_train1, cv=10, scoring='f1',verbose = 1)
    #[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    #[Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed:  2.2min finished
    metric.sort()
    metric_all['tree'] = metric[::-1]

    RF = RandomForestClassifier(n_estimators=30, criterion='gini', random_state=0)
    metric = cross_val_score(RF,trainX_all,trainy_all, cv=10, scoring='f1',verbose = 1)
    #[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    #[Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed:  4.6min finished
    metric.sort()
    metric_all['RF30'] = metric[::-1]

    RF = RandomForestClassifier(n_estimators=50, criterion='gini', random_state=0)
    metric = cross_val_score(RF,trainX_all,trainy_all, cv=10, scoring='f1',verbose = 1)
    #[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    #[Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed:  7.7min finished
    metric.sort()
    metric_all['RF50'] = metric[::-1]

    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    metric = cross_val_score(gnb, trainX_all,trainy_all, cv=10, scoring='f1',verbose = 1)
    #[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    #[Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed:   24.8s finished
    metric.sort()
    metric_all['gnb'] = metric[::-1]

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=None, priors=None)
    metric = cross_val_score(lda, trainX_all,trainy_all, cv=10, scoring='f1',verbose = 1)
    #[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    #[Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed:   42.5s finished
    metric.sort()
    metric_all['lda'] = metric[::-1]

    from sklearn.ensemble import GradientBoostingClassifier
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
    metric = cross_val_score(lda, trainX_all,trainy_all, cv=10, scoring='f1',verbose = 1)
    #[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    #[Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed:   39.8s finished
    metric.sort()
    metric_all['gb'] = metric[::-1]
    print(metric_all)
    metric_mean = metric_all.mean()
    print(metric_mean.sort_values(ascending=False))
    return metric_all
#————————————————————————————————————————
def triple():
    metric_all2 = pd.DataFrame()
    scoring = 'accuracy'
    from sklearn.model_selection import KFold
    kfold = KFold(n_splits=10, random_state=0)
    lr= LogisticRegression(C = 1.0,penalty = 'l2',solver = 'lbfgs')
    metric = cross_val_score(lr, trainX, trainY, cv=kfold, scoring=scoring)
    metric.sort()
    metric_all2['glm'] = metric[::-1]
    dt = DecisionTreeClassifier(criterion="entropy",max_depth=12)
    metric = cross_val_score(dt, trainX, trainY, cv=kfold, verbose = 1,scoring=scoring)
    metric.sort()
    metric_all2['dt'] = metric[::-1]
    X_train1, X_test1, y_train1, y_test1 = train_test_split(trainX,trainY, test_size = 0.5, random_state = 0)
    metric = cross_val_score(svc, X_train1, y_train1, cv=kfold, verbose = 1,scoring=scoring)
    metric.sort()
    metric_all2['svm'] = metric[::-1]
    gnb = GaussianNB()
    metric = cross_val_score(gnb, trainX, trainY, cv=kfold, verbose = 1,scoring=scoring)
    metric.sort()
    metric_all2['gnb'] = metric[::-1]
    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=None, priors=None)
    metric = cross_val_score(lda, trainX, trainY, cv=kfold, verbose = 1,scoring=scoring)
    metric.sort()
    metric_all2['LDA'] = metric[::-1]
    RF = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=0)
    metric = cross_val_score(RF, trainX, trainY, cv=kfold, verbose = 1,scoring=scoring)
    metric.sort()
    metric_all2['RF100'] = metric[::-1]
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
    metric = cross_val_score(RF, trainX, trainY, cv=kfold, verbose = 1,scoring=scoring)
    metric.sort()
    metric_all2['gb'] = metric[::-1]
    print(metric_all)
    metric_mean = metric_all.mean()
    print(metric_mean.sort_values(ascending=False))
    return metric_all2



def bi_percep():
    ppn = Perceptron(eta0=0.1, random_state=0)
    ppn.fit(X_train, y_train)
    y_pred = ppn.predict(X_test)
    acc1 = accuracy_score(y_test, y_pred)
    print(acc1)
    return ppn
    #0.9806563500533618

def train_and_test():
    result = pd.DataFrame()
    f = open(F_PKL, 'rb')
    testX_all = pickle.load(f)
    f.close()
    lr = RandomForestClassifier(n_estimators = 70,min_samples_split=100, min_samples_leaf=20,max_depth=8,max_features='sqrt',random_state=10)
    lr.fit(trainX_all,trainy_all)
    y_pred_binary = lr.predict(testX_all)
    y_pred_binary = list((np.array(y_pred_binary)-1)*(-1))
    result['binary'] = y_pred_binary
    stage2 = result[result['binary']==1].index.tolist()
    testX = []
    for i in stage2:
        textX.append(testX_all[i])
    gb = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.1,min_samples_split=300,max_features='sqrt',subsample=0.8,random_state=10)
    gb.fit(trainX,trainY)
    y_pred = list(gb.predict(testX))
    Stance = {0:'unrelated',1:'discuss',2:'agree',3:'disagree'}
    pred = []
    for i in range(len(y_pred_binary)):
        if y_pred_binary[i] == 0:
            pred.append('unrelated')
        else:
            pred.append(Stance[y_pred.pop(0)])
    dataframe = pd.read_hdf(F_H5)
    actual = list(dataframe['Stance'])
    report_score(actual,pred)
    return pred
pred = train_and_test()

    


