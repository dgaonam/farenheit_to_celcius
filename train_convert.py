from ast import Import
import tensorflow as tf
from keras.layers.core import Dense
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 4GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2158)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print("Error: -->" + e)

celsius = np.array([-40,-10,0,8,15,22,38],dtype=float)
fahrenheit = np.array([-40,14,32,46.4,59,71.6,100.4],dtype=float)

single_feature_normalizer = tf.keras.layers.Normalization(axis=None)
modelo = Sequential();
modelo.add(Dense(16,input_dim=1,name='Entrada'))
modelo.add(Dense(1,input_dim=1,name='Salida'))
modelo.add(single_feature_normalizer)

optimizer = Adam(learning_rate=0.01)

modelo.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['binary_accuracy'])

#modelo.compile(
#    optimizer = tf.keras.optimizers.Adam(0.1),
#    loss='mean_squared_error'
#)

print("Comienza el entrenamiento .....")
historial = modelo.fit(celsius,fahrenheit,epochs=3000,verbose=True)
print("Modelo Entrenado")

modelo.save("data/celsius_fahrenhenit.h5")

print("Hagamos un prediccion")

print("¿Cuales son los grados Celsius a convertir?")
c = float(input())
resultado = modelo.predict([[c]])
print("la conversion de celsius " + str(c) + " es  "+ str(resultado) + "fahren")

# evalua el modelo
scores = modelo.evaluate(celsius,fahrenheit)
print("\n%s: %.2f%%" % (modelo.metrics_names[1], scores[1]*100))