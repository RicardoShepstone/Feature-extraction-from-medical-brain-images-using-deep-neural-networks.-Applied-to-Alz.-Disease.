from matplotlib import pyplot as plt
import numpy as np
from sklearn import  metrics
import itertools
def generar_plot(historial,numero_fold,nombre_archivo):
    plt.plot(historial.history['loss'])
    plt.plot(historial.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('mean squared error')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.savefig(str(numero_fold)+nombre_archivo+'.png')
    plt.savefig(str(numero_fold)+nombre_archivo+'.pdf')

def representar_matrizconfusion(true_labels,predicted_labels,class_list,nombre_archivo):
    cm = metrics.confusion_matrix(true_labels, predicted_labels, labels=class_list)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    generar_matrizconfusion(cm, classes=class_list, normalize=False, title='Matriz de Confusión')
    plt.savefig(nombre_archivo+'matriz-confusion.png')

    # Plot normalized confusion matrix
    plt.figure()
    generar_matrizconfusion(cm, classes=class_list, normalize=True, title='Matriz de Confusión Normalizada')
    plt.savefig(nombre_archivo+'matriz-confusion-normalizada.png')
    
    
    
    
def generar_matrizconfusion(matriz_confusion, classes, normalize=False, title='Confusion matrix'):
    if normalize:
        matriz_confusion = matriz_confusion.astype('float') / matriz_confusion.sum(axis=1)[:, np.newaxis]
        print("Matriz de confusión normalizada")
    else:
        print('Matriz de confusión sin normalizar')

    plt.imshow(matriz_confusion, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, ['Normal','MCI','AD'], rotation=45)
    plt.yticks(tick_marks, ['Normal','MCI','AD'])

    fmt = '.3f' if normalize else 'd'
    umbral = matriz_confusion.max() / 2.

    for i, j in itertools.product(range(matriz_confusion.shape[0]), range(matriz_confusion.shape[1])):
        plt.text(j, i, format(matriz_confusion[i, j], fmt), horizontalalignment="center",
                 color="white" if matriz_confusion[i, j] > umbral else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')