import numpy as np
import pandas as pd
from pathlib import Path
import nibabel as nib
    
def cargar_datos(dataPath,CLF):

    dataPath='C:\\Users\\User\\Documents\\6\\TFG\\Imágenes\\muestras\\'
    print(dataPath)
    dataset=pd.read_excel(dataPath +'dataset.xls')
    df=pd.DataFrame(dataset)
    img_data=np.ndarray(shape=(362,157,189,156), dtype='float32')
    cantidad_datos=[0,0,0]
    if CLF==1:
        labels=np.ndarray(shape=(362,1), dtype='float32')
        x=0
        for file in Path(dataPath).glob('**/*.nii'):
            id=(str(file)[56:66])
            lista_estado=df[df['Subject ID']==id]['DX Group']
            estado=lista_estado.iloc[0]
            if estado=='Normal':
                labels[x]=[0]
                cantidad_datos[0]+=1
            elif estado=='AD':
                labels[x]=[2]
                cantidad_datos[2]+=1
            elif estado=='LMCI':
                labels[x]=[1]
                cantidad_datos[1]+=1
            elif estado=='MCI':
                labels[x]=[1]
                cantidad_datos[1]+=1
            img= nib.load(file)
            img_data[x,:,:,:]=img.get_fdata()
            print(str(x) + ' de 361')
            x +=1
    else:
        labels=np.ndarray(shape=(362,3), dtype='float32')
        x=0
        for file in Path(dataPath).glob('**/*.nii'):
            id=(str(file)[56:66])
            lista_estado=df[df['Subject ID']==id]['DX Group']
            estado=lista_estado.iloc[0]
            if estado=='Normal':
                labels[x,:]=[1,0,0]
            elif estado=='AD':
                labels[x,:]=[0,0,1]
            elif estado=='LMCI':
                labels[x,:]=[0,1,0]
            elif estado=='MCI':
                labels[x,:]=[0,1,0]
            img= nib.load(file)
            img_data[x,:,:,:]=img.get_fdata()
            print(str(x) + ' de 361')
            x +=1
    return labels,cantidad_datos

def procesado_datos(img_data, savePath):
    X=img_data[:,0:156,0:188,:]
    Y=np.ndarray(shape=(362,156,188,78), dtype='float32')
    Z=np.ndarray(shape=(362,156,94,78), dtype='float32')
    X_mean=np.ndarray(shape=(362,78,94,78), dtype='float32')
    for k in range(78):
        Y[:,:,:,k]=(X[:,:,:,2*k]+X[:,:,:,2*k+1])/2
        print('k='+str(k))
    for j in range(94):
        Z[:,:,j,:]=(Y[:,:,2*j,:]+Y[:,:,2*j+1,:])/2
        print('j='+str(j))
    for i in range(78):
        X_mean[:,i,:,:]=(Z[:,2*i,:,:]+Z[:,2*i+1,:,:])/2
        print('i='+str(i))
    np.save(Path(savePath), X_mean)

def clasificacion_binaria(input_data, labels, cantidad_datos, tipo_clasificacion):
    j=0
    if tipo_clasificacion == 'ADvsNORMAL':
        datos_extraidos=  np.ndarray(shape=(cantidad_datos[0]+cantidad_datos[2],78,94,78),dtype='float32')
        etiquetas_extraidas=np.empty(shape=(cantidad_datos[0]+cantidad_datos[2],1),dtype='float32')
        for i in range(len(labels)):
            if labels[i]==0:
                datos_extraidos[j]=input_data[i]
                etiquetas_extraidas[j]=labels[i]
                j+=1
            elif labels[i]==2:
                datos_extraidos[j]=input_data[i]
                etiquetas_extraidas[j]=labels[i]
                j+=1
        np.save(Path('C:\\Users\\User\\Documents\\6\\TFG\\datos procesados\\ADvsNORMAL'), datos_extraidos)
        np.save(Path('C:\\Users\\User\\Documents\\6\\TFG\\datos procesados\\ADvsNORMAL_labels'), etiquetas_extraidas)
    elif tipo_clasificacion== 'MCIvsNORMAL':
        datos_extraidos=  np.ndarray(shape=(cantidad_datos[0]+cantidad_datos[1],78,94,78),dtype='float32')
        etiquetas_extraidas=np.empty(shape=(cantidad_datos[0]+cantidad_datos[1],1),dtype='float32')
        for i in range(len(labels)):
            if labels[i]==0:
                datos_extraidos[j]=input_data[i]
                etiquetas_extraidas[j]=labels[i]
                j+=1
            elif labels[i]==1:
                datos_extraidos[j]=input_data[i]
                etiquetas_extraidas[j]=labels[i]
                j+=1
        np.save(Path('C:\\Users\\User\\Documents\\6\\TFG\\datos procesados\\MCIvsNORMAL'),datos_extraidos)
        np.save(Path('C:\\Users\\User\\Documents\\6\\TFG\\datos procesados\\MCIvsNORMAL_labels'), etiquetas_extraidas)
    elif tipo_clasificacion =='ADvsMCI':
        datos_extraidos=  np.ndarray(shape=(cantidad_datos[2]+cantidad_datos[1],78,94,78),dtype='float32')
        etiquetas_extraidas=np.empty(shape=(cantidad_datos[2]+cantidad_datos[1],1),dtype='float32')
        for i in range(len(labels)):
            if labels[i]==2:
                datos_extraidos[j]=input_data[i]
                etiquetas_extraidas[j]=labels[i]
                j+=1
            elif labels[i]==1:
                datos_extraidos[j]=input_data[i]
                etiquetas_extraidas[j]=labels[i]
                j+=1
        np.save(Path('C:\\Users\\User\\Documents\\6\\TFG\\datos procesados\\ADvsMCI'),datos_extraidos)
        np.save(Path('C:\\Users\\User\\Documents\\6\\TFG\\datos procesados\\ADvsMCI_labels'), etiquetas_extraidas)
