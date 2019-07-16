

```python
import sys
import numpy
import matplotlib
import pandas
import sklearn
```


```python
print('Python : {}'.format(sys.version))
print('Numpy : {}'.format(numpy.__version__))
print('Matplotlib : {}'.format(matplotlib.__version__))
print('Pandas : {}'.format(pandas.__version__))
print('sklearn : {}'.format(sklearn.__version__))
```

    Python : 3.6.5 |Anaconda, Inc.| (default, Mar 29 2018, 13:32:41) [MSC v.1900 64 bit (AMD64)]
    Numpy : 1.14.3
    Matplotlib : 2.2.2
    Pandas : 0.23.0
    sklearn : 0.19.1
    


```python
import numpy as np
from sklearn import preprocessing, cross_validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import pandas as pd
```


```python
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
names = ['id', 'clump_thickness', 'uniform_cell_size', 'uniform_cell_shape',
       'marginal_adhesion', 'single_epithelial_size', 'bare_nuclei',
       'bland_chromatin', 'normal_nucleoli', 'mitoses', 'class']
df = pd.read_csv(url, names=names)
```


```python
# Preprocessing the data
df.replace('?',-99999, inplace=True)
print(df.axes)
df.drop(['id'],1,inplace=True)
print(df.shape)
```

    [RangeIndex(start=0, stop=699, step=1), Index(['id', 'clump_thickness', 'uniform_cell_size', 'uniform_cell_shape',
           'marginal_adhesion', 'single_epithelial_size', 'bare_nuclei',
           'bland_chromatin', 'normal_nucleoli', 'mitoses', 'class'],
          dtype='object')]
    (699, 10)
    


```python
print(df.loc[90])
print(df.describe())
```

    clump_thickness           1
    uniform_cell_size         1
    uniform_cell_shape        1
    marginal_adhesion         1
    single_epithelial_size    2
    bare_nuclei               1
    bland_chromatin           3
    normal_nucleoli           1
    mitoses                   1
    class                     2
    Name: 90, dtype: object
           clump_thickness  uniform_cell_size  uniform_cell_shape  \
    count       699.000000         699.000000          699.000000   
    mean          4.417740           3.134478            3.207439   
    std           2.815741           3.051459            2.971913   
    min           1.000000           1.000000            1.000000   
    25%           2.000000           1.000000            1.000000   
    50%           4.000000           1.000000            1.000000   
    75%           6.000000           5.000000            5.000000   
    max          10.000000          10.000000           10.000000   
    
           marginal_adhesion  single_epithelial_size  bland_chromatin  \
    count         699.000000              699.000000       699.000000   
    mean            2.806867                3.216023         3.437768   
    std             2.855379                2.214300         2.438364   
    min             1.000000                1.000000         1.000000   
    25%             1.000000                2.000000         2.000000   
    50%             1.000000                2.000000         3.000000   
    75%             4.000000                4.000000         5.000000   
    max            10.000000               10.000000        10.000000   
    
           normal_nucleoli     mitoses       class  
    count       699.000000  699.000000  699.000000  
    mean          2.866953    1.589413    2.689557  
    std           3.053634    1.715078    0.951273  
    min           1.000000    1.000000    2.000000  
    25%           1.000000    1.000000    2.000000  
    50%           1.000000    1.000000    2.000000  
    75%           4.000000    1.000000    4.000000  
    max          10.000000   10.000000    4.000000  
    


```python
df.hist(figsize=(18,18))
plt.show()
```


![png](output_6_0.png)



```python
scatter_matrix(df,figsize=(18,18))
plt.show()
```


![png](output_7_0.png)



```python
# Training part
X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
```


```python
seed = 8
scoring = 'accuracy'
```


```python
# Defining the models to train
models = []
models.append(('KNN',KNeighborsClassifier(n_neighbors=5)))
models.append(('SVM',SVC()))

# Evaluate each model in turn
results = []
names = []
for name,model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results=model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
```

    KNN: 0.966006 (0.027028)
    SVM: 0.949903 (0.027447)
    


```python
# To make predictions on validation dataset
for name, model in models:
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(name)
    print(accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))
```

    KNN
    0.9714285714285714
                 precision    recall  f1-score   support
    
              2       0.97      0.99      0.98        92
              4       0.98      0.94      0.96        48
    
    avg / total       0.97      0.97      0.97       140
    
    SVM
    0.9785714285714285
                 precision    recall  f1-score   support
    
              2       1.00      0.97      0.98        92
              4       0.94      1.00      0.97        48
    
    avg / total       0.98      0.98      0.98       140
    
    


```python
clf =SVC()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)

example = np.array([[4,2,1,1,1,2,3,2,1]])
example = example.reshape(len(example),-1)
prediction = clf.predict(example)
print(prediction)
```

    0.9785714285714285
    [2]
    
