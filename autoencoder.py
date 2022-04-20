import tensorflow as tf
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow import keras
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import SGD

from keras.layers import Dense
from keras.models import Sequential
from keras import layers, Model, losses
from sklearn import preprocessing


class Autoencoder(Model):
  def __init__(self):
    super(Autoencoder, self).__init__()
    self.encoder = tf.keras.Sequential([
      #layers.Dense(15, input_shape=(train_x.shape[1],)),
      layers.Input(shape=(X_train.shape[1],)),
      layers.Dense(8, activation='relu'),
      layers.Dense(1, activation='relu')
    ])
    self.decoder = tf.keras.Sequential([
      #layers.Dense(1, activation='relu'),
      layers.Dense(8, activation='relu'),
      layers.Dense(15, activation='sigmoid')
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded




df = pd.read_csv(r'framin_Pure_Real_Titles.csv')


X = df.iloc[:,0:15]
y = df.iloc[:,-1]

labelencoder_Y = LabelEncoder()
y = labelencoder_Y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

sc = StandardScaler()
X_train = preprocessing.normalize(X_train)
X_test = preprocessing.normalize(X_test)
print(X_train[0])
autoencoder = Autoencoder()

autoencoder.compile(optimizer = 'rmsprop', loss = losses.BinaryCrossentropy(), metrics = ['accuracy'])

autoencoder.fit(X_train, X_train, batch_size = 1, epochs = 8)

#loss, acc = autoencoder.evaluate(X_test, y_test, verbose=0)
pred = autoencoder.encoder.predict(X_test)
pred_classes = autoencoder.encoder.predict_classes(X_test)

p=[]
for i in pred:
  for j in i:
    p.append(j)

f=0
r1=[]
r0=[]
print("class_pred","pred","labels")
for i in range(len(pred_classes)):
    print(p[i],"---",pred_classes[i],"---",y_test[i])
    if y_test[i]==1:
      r1.append(p[i])
    else:
      r0.append(p[i])

print("1:",max(r1),min(r1),sum(r1)/len(r1))
print("0:",max(r0),min(r0),sum(r0)/len(r0))






