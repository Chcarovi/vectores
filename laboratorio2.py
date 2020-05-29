# Se realiza la exportacion de las librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns



# creo dataset a trabajar
Datos = pd.read_csv('train.csv')
# creo una copia original para ver como estaba antes de trabajar con el original
Datos_copia = pd.read_csv('train.csv')

# cambio la posicion de la columna saliario de primeras para visualizarla mejor
cols = list(Datos.columns)
cols2 = cols[-1:] + cols[:-1]
Datos = Datos[cols2]

# Valido y lleno los datos vacios de cada columna con la media
Datos = Datos.fillna(Datos.mean())

numericos = Datos.fillna(Datos.mean()).describe()

# Creo las variables X y y
X = Datos.iloc[:, 1:].values
y = Datos.iloc[:, :1].values

# Correlacion
k = 5 
corrmat = Datos_copia.corr()
cols = corrmat.nlargest(k, 'price_range')['price_range'].index
cm = np.corrcoef(Datos_copia[cols].values.T)
sns.set(font_scale = 1.25)
hm = sns.heatmap(cm, cbar = True, annot = True, square = True, fmt = '.2f', annot_kws = {'size': 8}, yticklabels = cols.values, xticklabels = cols.values)
plt.show()


# particionar test y entrenamiento
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


from sklearn.svm import SVC
from sklearn import metrics


from sklearn.model_selection import cross_val_score
svc=SVC(kernel='linear', C=0.01).fit(X_train, y_train)
y_pred=svc.predict(X_test)
scores = cross_val_score(svc, X_train, y_train, cv=5, scoring='accuracy') #cv is cross validation
print()
print('Matrix Confusion LINEAL')
print('cross_val_score:', scores)
# puntaje_precisión
puntuacion = metrics.accuracy_score(y_test,y_pred)
print('puntaje_precisión: ', puntuacion)
titulo = 'Matrix Confusion LINEAL',  puntuacion
class_names = Datos.price_range
from sklearn.metrics import confusion_matrix
ax1= plt.subplot()
confm=confusion_matrix(y_test, y_pred)
print('Matriz confusion')
print(confm)
sns.heatmap(confm, cmap='Blues', annot=True, ax = ax1)
# labels, title and ticks
ax1.set_xlabel('Prediccion price_range');ax1.set_ylabel('Test price_range'); 
ax1.set_title(titulo); 
ax1.xaxis.set_ticklabels(['0', '1', '2', '3']); ax1.yaxis.set_ticklabels(['0', '1', '2', '3']);


from sklearn.model_selection import cross_val_score
svc=SVC(kernel='rbf', C=0.60).fit(X_train, y_train)
y_pred=svc.predict(X_test)
scores = cross_val_score(svc, X_train, y_train, cv=5, scoring='accuracy') #cv is cross validation
print()
print('Matrix Confusion Radial basis')
print('cross_val_score:', scores)
# puntaje_precisión
puntuacion = metrics.accuracy_score(y_test,y_pred)
print('puntaje_precisión: ', puntuacion)
titulo = 'Matrix Confusion Radial basis',  puntuacion
class_names = Datos.price_range
from sklearn.metrics import confusion_matrix
ax= plt.subplot()
confm=confusion_matrix(y_test, y_pred)
print('Matriz confusion')
print(confm)
sns.heatmap(confm, cmap='Blues', annot=True, ax = ax)
# labels, title and ticks
ax.set_xlabel('Prediccion price_range');ax.set_ylabel('Test price_range'); 
ax.set_title(titulo); 
ax.xaxis.set_ticklabels(['0', '1', '2', '3']); ax.yaxis.set_ticklabels(['0', '1', '2', '3']);


from sklearn.model_selection import cross_val_score
svc=SVC(kernel='poly', C=0.15).fit(X_train, y_train)
y_pred=svc.predict(X_test)
scores = cross_val_score(svc, X_train, y_train, cv=5, scoring='accuracy') #cv is cross validation
print()
print('Matrix Confusion Polynomial')
print('cross_val_score:', scores)
# puntaje_precisión
puntuacion = metrics.accuracy_score(y_test,y_pred)
print('puntaje_precisión: ', puntuacion)
titulo = 'Matrix Confusion Polynomial',  puntuacion
class_names = Datos.price_range
from sklearn.metrics import confusion_matrix
ax= plt.subplot()
confm=confusion_matrix(y_test, y_pred)
print('Matriz confusion')
print(confm)
sns.heatmap(confm, cmap='Blues', annot=True, ax = ax)
# labels, title and ticks
ax.set_xlabel('Prediccion price_range');ax.set_ylabel('Test price_range'); 
ax.set_title(titulo); 
ax.xaxis.set_ticklabels(['0', '1', '2', '3']); ax.yaxis.set_ticklabels(['0', '1', '2', '3']);




from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, nb_epoch=150, batch_size=10)
scores = model.evaluate(X, y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))






























