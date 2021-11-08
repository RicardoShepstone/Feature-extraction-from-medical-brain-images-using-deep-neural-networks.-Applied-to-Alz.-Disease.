import numpy as np
import pandas as pd
from pathlib import Path
import nibabel as nib
from cargayprocesado_datos import cargar_datos
from cargayprocesado_datos import procesado_datos
from cargayprocesado_datos import clasificacion_binaria
from matplotlib import pyplot as plt
import expresion_resultados
##########################################################

dataPath='C:\\Users\\User\\Documents\\6\\TFG\\Imágenes\\muestras\\'
#savePath='C:\\Users\\User\\Documents\\6\\TFG\\datos procesados\\img_data'
loadPath='C:\\Users\\User\\Documents\\6\\TFG\\datos procesados\\img_data.npy'

labels, cantidad_datos=cargar_datos(dataPath, 1)
#procesado_datos(img_data, savePath)

input_data= np.load(Path(loadPath))

########################################################################

compresion=15

input_data= input_data.reshape((362,78,94,78,1))



from keras import layers
from keras import models

import Autoencoders

encoder, decoder =Autoencoders.build_autoencoder_3(compresion)
inp=layers.Input(input_data.shape[1:])
code = encoder(inp)
reconstruction = decoder(code)

autoencoder = models.Model(inp,reconstruction)
autoencoder.summary()
autoencoder.compile(loss='MeanSquaredError', optimizer='Adam')

autoencoder.save_weights('autoencoder.h5')

from sklearn import  tree
from sklearn.model_selection import KFold


kfold=KFold(n_splits=5, shuffle=False)
score=[]
predicted_labels=np.array([])
true_labels=np.array([])
numero_fold=1
for train, test in kfold.split(input_data):
    print(f'entrenamiento fold: {numero_fold}')
    historial=autoencoder.fit(x=input_data[train], y=input_data[train], batch_size=2, epochs=20, verbose=1,validation_data=(input_data[test],input_data[test]))
    clf = tree.DecisionTreeClassifier()
    j=0
    k=0
    codigo_entrenamiento=np.empty(shape=(int(len(train)),compresion),dtype='float32')
    codigo_prueba=np.empty(shape=(int(len(test)),compresion),dtype='float32')
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
    
    
    predicted_labels=np.append(predicted_labels,prediction)
    true_labels=np.append(true_labels,labels_test)
    print('prediccion: '+ str(prediction))
    print('etiqueta: '+ str(labels_test))
    score.append(clf.score(codigo_prueba,labels_test))
    average=np.sum(score)/len(score)

    expresion_resultados.generar_plot(historial,numero_fold,'DecisionTree-Multiclass')
    autoencoder.load_weights('autoencoder.h5')  
    
    numero_fold +=1

expresion_resultados.representar_matrizconfusion(true_labels,predicted_labels,clf.classes_,'DecisionTree-Multiclass-')
