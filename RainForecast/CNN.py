import pandas as pd
import numpy as np
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv1D, Flatten, Dense, Dropout, MaxPooling1D
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.python.keras.metrics import AUC
warnings.filterwarnings('ignore')

test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')

train.fillna(train.mean(), inplace=True)
test.fillna(test.mean(), inplace=True)

train['hci'] = train['humidity'] * train['cloud']
train['hsi'] = train['humidity'] * train['sunshine']
train['csr'] = train['cloud'] / (train['sunshine'] + 1e-5)
train['rd'] = 100 - train['humidity']
train['sp'] = train['sunshine'] / (train['sunshine'] + train['cloud'] + 1e-5)
train['wi'] = (0.4 * train['humidity']) + (0.3 * train['cloud']) - (0.3 * train['sunshine'])

test['hci'] = test['humidity'] * test['cloud']
test['hsi'] = test['humidity'] * test['sunshine']
test['csr'] = test['cloud'] / (test['sunshine'] + 1e-5)
test['rd'] = 100 - test['humidity']
test['sp'] = test['sunshine'] / (test['sunshine'] + test['cloud'] + 1e-5)
test['wi'] = (0.4 * test['humidity']) + (0.3 * test['cloud']) - (0.3 * test['sunshine'])

X = train.drop(columns=['id', 'rainfall'])
y = train['rainfall']
X_test = test.drop(columns=['id'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=32, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='tanh') 
])

from tensorflow.python.keras.optimizer_v1 import SGD

optimizer = SGD(lr=0.001, momentum=0.9, decay=1e-6)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[AUC(name='auc')])

early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-5, verbose=1)

history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

test_preds = model.predict(X_test_scaled).flatten()

if np.isnan(test_preds).sum() > 0:
    print(f"Found {np.isnan(test_preds).sum()} NaN values in predictions. Fixing them...")
    test_preds = np.nan_to_num(test_preds) 
submission = pd.DataFrame({"id": test['id'], "rainfall": test_preds})
submission.to_csv("submission.csv", index=False)