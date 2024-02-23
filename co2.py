import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

emissiondata = pd.read_csv('CarbonEmission.csv')

#X=emissiondata[['Days','CO2Emission']].values
#y=emissiondata['Category'].values

print('tp1')
X=emissiondata[['Days','CO2Emission']].to_numpy()
y=emissiondata['Category'].to_numpy()
print('tp2')


h = 0.02
C = 1.0
print('tp3')

svc = svm.SVC(kernel='linear', C=C).fit(X,y)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=1).fit(X,y)
poly_svc = svm.SVC(kernel='poly', degree=3, C=1).fit(X,y)
print('tp4')


#x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
#y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1

x_min = np.array(X[:,0].tolist(), dtype=float).min() - 1
x_max = np.array(X[:,0].tolist(), dtype=float).max() + 1
y_min = np.array(X[:,1].tolist(), dtype=float).min() - 1
y_max = np.array(X[:,1].tolist(), dtype=float).max() + 1
print('tp5')


xx,yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
print('tp6')

titles = ['SVC with linear kernel', 'SVC with RBF kernel', 'SVC with polynomial kernel']

for i, clf in enumerate((svc, rbf_svc, poly_svc)):
    plt.figure(figsize=(14,10))
    plt.subplot(2,2,i+1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.coolwarm)
    plt.xlabel('Days')
    plt.ylabel('Emission')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks()
    plt.yticks()
    plt.title(titles[i])

plt.show()