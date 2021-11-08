import keras

from keras import layers
from keras import models

def build_autoencoder_1(capa_intermedia):
     #encoder
    
    encoder=models.Sequential()
    encoder.add(layers.Conv3D(32,3,1,activation='relu',input_shape=(78,94,78,1)))
    encoder.add(layers.MaxPooling3D(2))
    encoder.add(layers.BatchNormalization())
    encoder.add(layers.Conv3D(32,3,1,activation='relu'))
    encoder.add(layers.MaxPooling3D(2))
    encoder.add(layers.BatchNormalization())
    encoder.add(layers.Conv3D(64,3,1,activation='relu'))
    encoder.add(layers.MaxPooling3D(2))
    encoder.add(layers.BatchNormalization())
    encoder.add(layers.Conv3D(64,3,1,activation='relu'))
    encoder.add(layers.Flatten())
    encoder.add(layers.Dense(capa_intermedia,activation='relu'))

    
    #decoder
    
    decoder=models.Sequential()
    decoder.add(layers.Dense(18432,activation='relu'))
    decoder.add(layers.Reshape((6,8,6,64), input_shape=(18432,)))
    decoder.add(layers.Conv3DTranspose(64,3,1,activation='relu'))
    decoder.add(layers.UpSampling3D(2))
    decoder.add(layers.BatchNormalization())
    decoder.add(layers.Conv3DTranspose(32,3,1,activation='relu'))
    decoder.add(layers.UpSampling3D(2))
    decoder.add(layers.BatchNormalization())
    decoder.add(layers.Conv3DTranspose(32,3,1,activation='relu'))
    decoder.add(layers.UpSampling3D(2))
    decoder.add(layers.BatchNormalization())
    decoder.add(layers.Conv3DTranspose(1,3,1,activation='relu'))
    
    return encoder, decoder


def build_autoencoder_2(capa_intermedia):
    
    #encoder
    
    encoder =models.Sequential()
    encoder.add(layers.Conv3D(64,(3,5,3),(3,1,3),activation='relu',input_shape=(78,94,78,1)))
    encoder.add(layers.BatchNormalization())
    encoder.add(layers.Conv3D(64,(3,5,3),(1,5,1),activation='relu'))
    encoder.add(layers.BatchNormalization())
    encoder.add(layers.Conv3D(128,3,3,activation='relu'))
    encoder.add(layers.BatchNormalization())
    encoder.add(layers.Conv3D(128,2,2,activation='relu'))
    encoder.add(layers.BatchNormalization())
    encoder.add(layers.Flatten())
    encoder.add(layers.Dense(capa_intermedia))
    
    #decoder
    
    decoder=models.Sequential()
    decoder.add(layers.Dense(6144,activation='relu'))
    decoder.add(layers.Reshape((4,3,4,128), input_shape=(6144,)))
    decoder.add(layers.BatchNormalization())
    decoder.add(layers.Conv3DTranspose(128,2,2,activation='relu'))
    decoder.add(layers.BatchNormalization())
    decoder.add(layers.Conv3DTranspose(64,3,3,activation='relu'))
    decoder.add(layers.BatchNormalization())
    decoder.add(layers.Conv3DTranspose(64,(3,5,3),(1,5,1),activation='relu'))
    decoder.add(layers.BatchNormalization())
    decoder.add(layers.Conv3DTranspose(1,(3,5,3),(3,1,3),activation='relu'))
    
    return encoder, decoder

def build_autoencoder_3(capa_intermedia):
    
    #encoder
    
    encoder=models.Sequential()
    encoder.add(layers.Conv3D(64,3,1,activation='relu',padding='same',input_shape=(78,94,78,1)))
    encoder.add(layers.Conv3D(64,(3,5,3),(3,1,3),activation='relu'))
    encoder.add(layers.BatchNormalization())
    encoder.add(layers.Conv3D(64,3,1,activation='relu',padding='same'))
    encoder.add(layers.Conv3D(64,(3,5,3),(1,5,1),activation='relu'))
    encoder.add(layers.BatchNormalization())
    encoder.add(layers.Conv3D(64,3,1,activation='relu',padding='same'))
    encoder.add(layers.Conv3D(64,3,3,activation='relu'))
    encoder.add(layers.BatchNormalization())
    encoder.add(layers.Conv3D(64,3,1,activation='relu',padding='same'))
    encoder.add(layers.Conv3D(64,2,2,activation='relu'))
    encoder.add(layers.BatchNormalization())
    encoder.add(layers.Flatten())
    encoder.add(layers.Dense(capa_intermedia))
    
    #decoder
    
    decoder=models.Sequential()
    decoder.add(layers.Dense(3072,activation='relu'))
    decoder.add(layers.Reshape((4,3,4,64), input_shape=(3072,)))
    decoder.add(layers.BatchNormalization())
    decoder.add(layers.Conv3DTranspose(64,2,2,activation='relu'))
    decoder.add(layers.Conv3D(64,3,1,activation='relu',padding='same'))
    decoder.add(layers.BatchNormalization())
    decoder.add(layers.Conv3DTranspose(64,3,3,activation='relu'))
    decoder.add(layers.Conv3D(64,3,1,activation='relu',padding='same'))
    decoder.add(layers.BatchNormalization())
    decoder.add(layers.Conv3DTranspose(64,(3,5,3),(1,5,1),activation='relu'))
    decoder.add(layers.Conv3D(64,3,1,activation='relu',padding='same'))
    decoder.add(layers.BatchNormalization())
    decoder.add(layers.Conv3DTranspose(1,(3,5,3),(3,1,3),activation='relu'))
    
    return encoder, decoder