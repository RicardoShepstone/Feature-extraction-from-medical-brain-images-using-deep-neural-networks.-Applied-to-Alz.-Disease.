from matplotlib import pyplot as plt

def generar_plot(historial,numero_fold,nombre_archivo):
    plt.plot(historial.history['loss'])
    plt.plot(historial.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('mean squared error')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.savefig(str(numero_fold)+nombre_archivo+'.png')
    plt.savefig(str(numero_fold)+nombre_archivo+'.pdf')


