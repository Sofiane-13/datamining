import pandas
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from mca2 import *
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
df = pandas.read_csv("german.data",sep="\t")
#print(df.shape)
#print(df.describe(include='all'))
#df.purpose.value_counts().sort_index().plot.pie()
#df.duration.plot.hist()
#df.duration.plot.box()
#pyplot.show()
dfbin=pandas.get_dummies(df.iloc[:,:20])
print(df.iloc[:,:20])
dfbin.shape
#print(dfbin.shape)
dfmca=mca(dfbin,benzecri=False)
nouvellesCordonnées=(dfmca.fs_r(N=61))
nouvellesCordonnées.shape
#print(nouvellesCordonnées.shape)
classe=df.iloc[:,20]
#print(classe)
# Clf= tree.DecisionTreeClassifier()
Clf= KNeighborsClassifier(n_neighbors=15)
#Clf= Clf.fit(nouvellesCordonnées,classe)
#print(Clf)
# dotfile = open("./monArbre.dot",'w')
# tree.export_graphviz(Clf,
#                         filled=True, rounded=True,
#                         special_characters=True , out_file= dotfile)
# dotfile.close()
# http://scikit-learn.org/stable/modules/cross_validation.html 

X_train, X_test, y_train, y_test = train_test_split(nouvellesCordonnées, classe,test_size=0.4)

monmodel = Clf.fit(X_train,y_train)
# score = Clf.score(X_test,y_test)
monScore=monmodel.predict_proba(X_test)
print(monScore)
fpr, tpr, thresholds = roc_curve(y_test,monScore[:,1], pos_label=1)
# print(confusion_matrix(y_test,predit))
auc = metrics.auc(fpr, tpr)
print(auc)
pyplot.plot(fpr,tpr,lw=1,alpha=0.3)
pyplot.show()