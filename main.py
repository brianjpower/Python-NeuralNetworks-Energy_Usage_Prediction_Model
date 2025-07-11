
import pandas as pd
import numpy as np
#import plot.matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
#from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input
import gc
from tensorflow.keras.backend import clear_session



from sklearn.preprocessing import StandardScaler
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from sklearn.model_selection import train_test_split



df = pd.read_csv('data_energy_consumption.csv')
pd.set_option('display.max_columns', None)
print(df.describe())
#print(df.head())
#print(df.tail())
#print(df['Appliances'].unique())

#Format the date and time data
df['date'] = pd.to_datetime(df['date'])
#df['day'] = df.date.dt.day
df['month'] = df.date.dt.month
df['hour'] = df.date.dt.hour
print(df.head())

#set y to the appliance data and convert high to 1 and normal to zero
y = np.where(df['Appliances']=="normal", 0, 1)


#Remove the Appliances columns from the feature dataset since this will be
#target categorical variable
x = df.drop(columns=['date','Appliances'])
#x = df.drop(columns=['date'])
print(x.head())
#Now we want to scale the x data
scaler = StandardScaler()
x = scaler.fit_transform(x) # this is setting mean to zero and std dev to 1 bascially, shortcut for x=(x-x.mean())/x.std()

V = x.shape[1] # store the number of predictors/features
print(V)
# one hot encoding for y categorical data, enable nn to handle multi-class classification
y = to_categorical(y)
K = y.shape[1] # store the number of classes
print(K)

model = Sequential()

# Add layers to the model
#model.add(Dense(units=64, activation='relu', input_shape=(10,)))
#model.add(Dense(units=1, activation='sigmoid'))
model = Sequential()
model.add(Input(shape=(V,)))  # Explicitly define the input shape
model.add(Dense(units=64, activation='relu'))  # Add subsequent layers
model.add(Dense(units=K, activation='softmax'))

# Compile the model
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer=SGD(), loss='binary_crossentropy', metrics=['accuracy'])


# Summary of the model
#model.summary()

# Save initial set of weights
w = model.get_weights()

# Train and test data split with 70/30 split
TOT = x.shape[0]
N = int(TOT * 0.70)
M = int(TOT * 0.30)
B = 2  # Number of replications for bootstrapping

acc_train = np.empty(B)  # accumulated train and test data for 20 bootstraps
acc_test = np.empty(B)

for b in range(B):
    #randomly choose train indices and set test indices to complement
    train_indices = np.random.choice(range(TOT), N, replace=False)
    test_indices = np.setdiff1d(range(TOT), train_indices)

    x_train, y_train = x[train_indices], y[train_indices]
    x_test, y_test = x[test_indices], y[test_indices]

    model.set_weights(w)  # Reset weights

    fit = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        verbose=0
    )
    clear_session()
    gc.collect()
    n_epoch = len(fit.history['accuracy'])
    acc_train[b] = fit.history['accuracy'][-1]
    acc_test[b] = fit.history['val_accuracy'][-1]


#plt.boxplot([acc_train, acc_test], labels=['Train', 'Test'])
#plt.ylabel("Accuracy")
#plt.show()

# Hyperparameter tuning
#H_vec = [0, 5, 10, 15, 20, 25]
H_vec = [15,20,25]

H = len(H_vec)
L = round(TOT * 0.2)
test_indices = np.random.choice(range(TOT), L, replace=False)
x_test, y_test = x[test_indices], y[test_indices]

set_indices = np.setdiff1d(range(TOT), test_indices)
N = round((TOT - L) * 0.8)
M = TOT - L - N

acc_train = np.empty((B, H))
acc_val = np.empty((B, H))

train_indices = np.random.choice(set_indices, N, replace=False)
val_indices = np.setdiff1d(set_indices, train_indices)

x_train, y_train = x[train_indices], y[train_indices]
x_val, y_val = x[val_indices], y[val_indices]

"""
for h, my_units in enumerate(H_vec):
    model=Sequential()
    model.add(Input(shape=(V,)))  # Explicitly define the input shape
    model.add(Dense(my_units, activation='relu'))  # Add subsequent layers
    model.add(Dense(units=K, activation='softmax'))
    #model = Sequential([
    #Dense(units, activation='relu', input_shape=(V,)),
    #Dense(K, activation='softmax')
    #])
    #model.add(Input(shape=(V,)))  # Explicitly define the input shape
    #model.add(Dense(units=64, activation='relu'))  # Add subsequent layers
    #model.add(Dense(units=K, activation='softmax'))

    model.compile(
        loss="binary_crossentropy",
        optimizer=SGD(),
        metrics=["accuracy"]
    )

    fit = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        verbose=0
    )

    n_epoch = len(fit.history['accuracy'])
    acc_train[b, h] = fit.history['accuracy'][-1]
    acc_val[b, h] = fit.history['val_accuracy'][-1]
    clear_session()
    gc.collect()

    print(n_epoch)
    print(acc_train[b, h])
    print(acc_val[b, h])

mean_acc_train = np.mean(acc_train, axis=0)
mean_acc_val = np.mean(acc_val, axis=0)
print(mean_acc_train)
print(mean_acc_val)
# Plot training and validation accuracy
plt.plot(H_vec, mean_acc_train, label='Training Accuracy', color='black', linewidth=2)
plt.plot(H_vec, mean_acc_val, label='Validation Accuracy', color='darkorange', linewidth=2)
plt.xlabel("H")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
"""



for b in range(B):
    train_indices = np.random.choice(set_indices, N, replace=False)
    val_indices = np.setdiff1d(set_indices, train_indices)

    x_train, y_train = x[train_indices], y[train_indices]
    x_val, y_val = x[val_indices], y[val_indices]

    for h, my_units in enumerate(H_vec):
        model=Sequential()
        model.add(Input(shape=(V,)))  # Explicitly define the input shape
        model.add(Dense(my_units, activation='relu'))  # Add subsequent layers
        model.add(Dense(units=K, activation='softmax'))
        #model = Sequential([
            #Dense(units, activation='relu', input_shape=(V,)),
            #Dense(K, activation='softmax')
        #])
        #model.add(Input(shape=(V,)))  # Explicitly define the input shape
        #model.add(Dense(units=64, activation='relu'))  # Add subsequent layers
        #model.add(Dense(units=K, activation='softmax'))

        model.compile(
            loss="binary_crossentropy",
            optimizer=SGD(),
            metrics=["accuracy"]
        )

        fit = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            verbose=0
        )

        n_epoch = len(fit.history['accuracy'])
        acc_train[b, h] = fit.history['accuracy'][-1]
        acc_val[b, h] = fit.history['val_accuracy'][-1]
        clear_session()
        gc.collect()

        print(n_epoch)
        print(acc_train[b, h])
        print(acc_val[b, h])

mean_acc_train = np.mean(acc_train, axis=0)
mean_acc_val = np.mean(acc_val, axis=0)
print(mean_acc_train)
print(mean_acc_val)
# Plot training and validation accuracy
plt.plot(H_vec, mean_acc_train, label='Training Accuracy', color='black', linewidth=2)
plt.plot(H_vec, mean_acc_val, label='Validation Accuracy', color='darkorange', linewidth=2)
plt.xlabel("H")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Plot of training & val accuracy with at various hyperparameters")
plt.show()


"""
# Plot training and validation accuracy
plt.plot(H_vec, mean_acc_train, label='Training Accuracy', color='black', linewidth=2)
plt.plot(H_vec, mean_acc_val, label='Validation Accuracy', color='darkorange', linewidth=2)
plt.xlabel("H")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
"""
# Best hyperparameter
H_best = H_vec[np.argmax(mean_acc_val)]
print("Best H:", H_best)

# Final model training
test_mask = np.ones(TOT, dtype=bool)
test_mask[test_indices] = False

model = Sequential([
    Dense(H_best, activation='relu', input_shape=(V,)),
    Dense(K, activation='sigmoid')
])

model.compile(
    loss="binary_crossentropy",
    optimizer=SGD(),
    metrics=["accuracy"]
)

fit = model.fit(
    x[test_mask], y[test_mask],
    validation_data=(x_test, y_test),
    verbose=0
)

# Print metrics
print(fit.history)
#Save the final energy model
from tensorflow.keras.models import load_model

model.save('final_energy_model.keras')
model = load_model('final_energy_model.keras')

#Now take a new set of data and predict the energy consumption with the final model




# Read in the original dataset
df = pd.read_csv('data_energy_consumption.csv')

# Step 1: Create a new dataset with 10% of the rows from the original dataset
new_df = df.sample(frac=0.1, random_state=42)

# Step 2: Save the new dataset (optional, to verify or for future use)
new_df.to_csv('new_energy_consumption_data.csv', index=False)


# Load new data
new_data = pd.read_csv('new_energy_consumption_data.csv')

# Perform the same feature engineering
new_data['date'] = pd.to_datetime(new_data['date'])
new_data['month'] = new_data.date.dt.month
new_data['hour'] = new_data.date.dt.hour

# Drop unnecessary columns
new_x = new_data.drop(columns=['date', 'Appliances'])

# Scale the new data using the same scaler
new_x = scaler.transform(new_x)  # Use the same scalers from

# Make predictions on the new data
predictions = model.predict(new_x)
predicted_classes = np.argmax(predictions, axis=1)
print(predicted_classes)
 #Finally, visualise the final model
#from tensorflow.keras.utils import plot_model
#plot_model(model, to_file='model_structure.png', show_shapes=True, show_layer_names=True)
print("Model Summary:")
model.summary()


