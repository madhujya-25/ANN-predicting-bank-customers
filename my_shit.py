import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


import keras
from keras.models import Sequential
from keras.layers import Dense



#classifier = Sequential()
#classifier.add(Dense(output_dim = 6, input_dim = 11, activation = "tanh", init = "uniform")) 
#classifier.add(Dense(output_dim = 6, activation = "tanh", init = "uniform"))
#classifier.add(Dense(output_dim = 1, activation = "sigmoid", init = "uniform"))
#classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
#classifier.fit(X_train,y_train, batch_size = 10, epochs = 500)


#y_pred = classifier.predict(X_test) 
#y_pred = (y_pred > 0.5)  

 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)





from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier(my_opti, inp, acti):
    classifier = Sequential()
    classifier.add(Dense(output_dim = inp, input_dim = 11, activation = acti, init = "uniform")) 
    classifier.add(Dense(output_dim = inp, activation = acti, init = "uniform"))
    classifier.add(Dense(output_dim = 1, activation = "sigmoid", init = "uniform"))
    classifier.compile(optimizer = my_opti, loss = "binary_crossentropy", metrics = ["accuracy"])
    return classifier 




classifier = KerasClassifier(build_fn = build_classifier, epochs = 500)

parameters = {'batch_size': [10, 25],
              'my_opti': ['adam', 'rmsprop'],
              'inp': [6, 12, 24],
              'acti': ['relu', 'tanh']}

from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_par = grid_search.best_params_
best_acc = grid_search.best_score_


## best parameters after running grid search     
#classifier = KerasClassifier(build_fn = build_classifier,batch_size = 10, epochs = 500)



from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()




















