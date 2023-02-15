import numpy as np
from dTree import DecisionTree


X_train = np.loadtxt("D3leaves.txt", usecols=(0,1), dtype=float)
Y_train = np.loadtxt("D3leaves.txt", usecols=2, dtype=int)
print(X_train)
print(Y_train)
clf = DecisionTree()


#clf = DecisionTreeClassifier(criterion='entropy',random_state=1234,decimals=0)
clf.fit(X_train,Y_train)
#text_representation = tree.export_text(clf)
#print(text_representation)








