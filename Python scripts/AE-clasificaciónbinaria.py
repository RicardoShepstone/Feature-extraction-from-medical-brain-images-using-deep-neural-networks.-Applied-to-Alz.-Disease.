import numpy as np
import pandas as pd
from pathlib import Path
import nibabel as nib
from cargayprocesado_datos import cargar_datos
from cargayprocesado_datos import procesado_datos
from cargayprocesado_datos import clasificacion_binaria
##########################################################

#comentar y adaptar

dataPath='C:\\Users\\User\\Documents\\6\\TFG\\Imágenes\\muestras\\'
#savePath='C:\\Users\\User\\Documents\\6\\TFG\\datos procesados\\img_data'
loadPath='C:\\Users\\User\\Documents\\6\\TFG\\datos procesados\\img_data.npy'
labels, cantidad_datos=cargar_datos(dataPath, 1)
#procesado_datos(img_data, savePath)

input_data= np.load(Path(loadPath))
#clasificacion_binaria(input_data, labels, cantidad_datos, tipo_clasificacion)

########################################################################

input_data= input_data.reshape((362,78,94,78,1))


import keras

from keras import layers
from keras import models

from Autoencoders import build_autoencoder_1

encoder, decoder =build_autoencoder_1(50)
inp=layers.Input(input_data.shape[1:])
code = encoder(inp)
reconstruction = decoder(code)

autoencoder = models.Model(inp,reconstruction)
autoencoder.summary()
autoencoder.compile(loss='MeanSquaredError', optimizer='Adam')

autoencoder.save_weights('autoencoder.h5')

from sklearn import datasets, tree, metrics
from sklearn.model_selection import KFold


kfold=KFold(n_splits=5, shuffle=False)
score=[]
precision_fold = []
loss_fold = []

numero_fold=1
for train, test in kfold.split(input_data):
    print(f'entrenamiento fold: {numero_fold}')
    entrenamiento=autoencoder.fit(x=input_data[train], y=input_data[train], batch_size=2, epochs=20, verbose=1,validation_data=(input_data[test],input_data[test]))
    clf = tree.DecisionTreeClassifier()
    j=0
    k=0
    codigo_entrenamiento=np.empty(shape=(int(len(train)),50),dtype='float32')
    codigo_prueba=np.empty(shape=(int(len(test)),50),dtype='float32')
    labels_train=np.empty(shape=(int(len(train)),1),dtype='float32')
    labels_test=np.empty(shape=(int(len(test)),1),dtype='float32')
    for i in train:
        img_train=input_data[i]
        codigo_entrenamiento[j]=encoder.predict(img_train[None])[0]
        labels_train[j]=labels[i]
        j+=1
        
    print(codigo_entrenamiento.shape)
    clf.fit(codigo_entrenamiento, labels_train)
    
    for i in test:
        img_test=input_data[i]
        codigo_prueba[k]=encoder.predict(img_test[None])[0]
        labels_test[k]=labels[i]
        k+=1
    prediction=clf.predict(codigo_prueba)
    
    print('prediccion: '+ str(prediction))
    print('etiqueta: '+ str(labels_test))
    score.append(clf.score(codigo_prueba,labels_test))
    average=np.sum(score)/len(score)
    autoencoder.load_weights('autoencoder.h5')  

    numero_fold +=1

print('------------------------------------------------------------------------')
for i in range(0, len(precision_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_fold[i]} - Accuracy: {precision_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(precision_fold)} (+- {np.std(precision_fold)})')
print(f'> Loss: {np.mean(loss_fold)}')
print('------------------------------------------------------------------------')
