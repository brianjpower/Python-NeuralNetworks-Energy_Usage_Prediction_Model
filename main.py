
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
#from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



df = pd.read_csv('data_energy_consumption.csv')
pd.set_option('display.max_columns', None)
print(df.describe())
print(df.head())
print(df.tail())
print(df['Appliances'].unique())

#Format the date and time data
df['date'] = pd.to_datetime(df['date'])
df['month'] = df.date.dt.month
df['hour'] = df.date.dt.hour


#Remove the Appliances columns from the feature dataset since this will be
#target categorical variable
x = df.drop(columns=['Appliances'])
y = df['Appliances']
print(x.head())
print(x.tail())
print(y.head())

model = Sequential()

# Add layers to the model
#model.add(Dense(units=64, activation='relu', input_shape=(10,)))
#model.add(Dense(units=1, activation='sigmoid'))
model = Sequential()
model.add(Input(shape=(10,)))  # Explicitly define the input shape
model.add(Dense(units=64, activation='relu'))  # Add subsequent layers
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()