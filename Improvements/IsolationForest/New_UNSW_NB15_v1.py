import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, BatchNormalization, Multiply
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn.compose import ColumnTransformer, make_column_selector
import optuna
import tensorflow as tf
from keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import json
import os
import pickle

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

RAND = 1337
np.random.seed(RAND)

DATA_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__"in locals() else "./"

train = pd.read_csv(os.path.join(DATA_DIR, 'UNSW_NB15_training-set.csv'))
test = pd.read_csv(os.path.join(DATA_DIR, 'UNSW_NB15_testing-set.csv'))

logging.info("Data Preprocessing...")

train.head()

cols_to_drop = ["id", "attack_cat"]
train = train.drop(columns=cols_to_drop, axis=1)
test = test.drop(columns=cols_to_drop, axis=1)

X_train_val_df = train.drop("label", axis=1)
y_train_val_df = train["label"]
y_test_df = test["label"]
X_test_df = test.drop("label", axis=1)

X_train_df, X_val_df, y_train_df, y_val_df = train_test_split(
    X_train_val_df, y_train_val_df, test_size=0.2, random_state=RAND)

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'),
         make_column_selector(dtype_include=['object', 'category'])),

        ('num', StandardScaler(),
         make_column_selector(dtype_include=['number']))
    ])

os.makedirs('./models', exist_ok=True)
with open(os.path.join(DATA_DIR, 'models', 'preprocessor_v2.pkl'), 'wb') as f:
    pickle.dump(preprocessor, f)

X_train = preprocessor.fit_transform(X_train_df)
X_val = preprocessor.transform(X_val_df)
X_test = preprocessor.transform(X_test_df)

X_train = X_train.toarray()
X_val = X_val.toarray()
X_test = X_test.toarray()
y_train = np.array(y_train_df)
y_val = np.array(y_val_df)
y_test = np.array(y_test_df)

print("Train Infs:", np.isinf(X_train).sum())
print("Val Infs:", np.isinf(X_val).sum())
print("Test Infs:",  np.isinf(X_test).sum())
print("Max:", X_train.max(), "Min:", X_train.min())
print("Label values:", np.unique(y_train))

logging.info(f"X_train shape: {X_train.shape}")

# ── Isolation Forest
logging.info("Starting Isolation Forest hyperparameter optimization")

def objective_iso(trial):
    contamination = trial.suggest_float("contamination", 0.01, 0.5)

    iso = IsolationForest(contamination=contamination, random_state=RAND, n_jobs=-1)
    iso.fit(X_train[y_train == 0])  # treina só com tráfego benigno

    anomaly_pred = iso.predict(X_val)
    y_iso_pred = (anomaly_pred == -1).astype(int)  # -1 → ataque, 1 → normal

    return f1_score(y_val, y_iso_pred, average='binary')

study_iso = optuna.create_study(direction='maximize')
study_iso.optimize(objective_iso, n_trials=20)

best_contamination = study_iso.best_params['contamination']
logging.info(f"Best contamination: {best_contamination:.4f}")

BATCH_SIZE=1024
EPOCHS=1000

classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, weights))
logging.info(f"Class weights: {class_weight_dict}")

logging.info("Starting hyperparameter optimization")

strategy = tf.distribute.MirroredStrategy()

def objective(trial):
  with strategy.scope():
    inputs = tf.keras.Input(shape=(X_train.shape[1],))

    x = Dense(1024, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Dense(768, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)

    # Camada de atenção
    attention = Dense(512, activation='softmax')(x)
    x = Multiply()([x, attention])

    output = Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=output)

    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

    model.compile(loss='binary_crossentropy',optimizer=Adam(learning_rate=lr),metrics=['accuracy'])

  early_stop = EarlyStopping(
      monitor='val_loss',
      patience=10,
      restore_best_weights=True
  )


  model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=1000,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    class_weight=class_weight_dict,
    verbose=0
  )


  loss, accuracy = model.evaluate(X_val, y_val, verbose=0)

  return accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

logging.info("Best hyperparameters:")
for key, value in study.best_params.items():
  logging.info(f"{key}: {value}")

best_lr = study.best_params['lr']

checkpoint = ModelCheckpoint(
    os.path.join(DATA_DIR, 'checkpoints', 'checkpoint.keras'),
    monitor='val_loss',
    save_best_only=True
)

from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam

with strategy.scope():
  inputs = tf.keras.Input(shape=(X_train.shape[1],))

  x = Dense(1024, activation='relu')(inputs)
  x = BatchNormalization()(x)
  x = Dropout(0.2)(x)

  x = Dense(768, activation='relu')(x)
  x = BatchNormalization()(x)
  x = Dropout(0.2)(x)

  x = Dense(512, activation='relu')(x)
  x = Dropout(0.2)(x)

  # Camada de atenção
  attention = Dense(512, activation='softmax')(x)
  x = Multiply()([x, attention])

  output = Dense(1, activation='sigmoid')(x)

  model = tf.keras.Model(inputs=inputs, outputs=output)

  model.summary()

  model.compile(loss='binary_crossentropy',optimizer=Adam(learning_rate=best_lr),metrics=['accuracy'])

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=100,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    class_weight=class_weight_dict,
    verbose=0
)
model.save(os.path.join(DATA_DIR, 'models', 'model_new_unsw_nb15_v2.keras'))

# Pipeline combinado: Isolation Forest → DNN
iso_final = IsolationForest(contamination=best_contamination, random_state=RAND, n_jobs=-1)
iso_final.fit(X_train[y_train == 0])

os.makedirs(os.path.join(DATA_DIR, 'models'), exist_ok=True)
with open(os.path.join(DATA_DIR, 'models', 'model_new_unsw-nb15-v1_iso.pkl'), 'wb') as f:
    pickle.dump(iso_final, f)

anomaly_pred = iso_final.predict(X_test)
X_suspicious = X_test[anomaly_pred == -1]

y_dnn_pred = (model.predict(X_suspicious) > 0.5).astype(int).flatten()

y_pred = np.zeros(len(X_test), dtype=int)
y_pred[anomaly_pred == -1] = y_dnn_pred

logging.info(f"Isolation Forest flagged {(anomaly_pred == -1).sum()} / {len(X_test)} samples as suspicious")

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred , average="binary")
precision = precision_score(y_test, y_pred , average="binary")
f1 = f1_score(y_test, y_pred, average="binary")
print("----------------------------------------------")
print("accuracy")
print("%.3f" %accuracy)
print("recall")
print("%.3f" %recall)
print("precision")
print("%.3f" %precision)
print("f1score")
print("%.3f" %f1)

results = {
    "accuracy": float(accuracy),
    "recall": float(recall),
    "precision": float(precision),
    "f1": float(f1)
}
with open(os.path.join(DATA_DIR, 'models', 'results_model_new_unsw_nb15-v2.json'), 'w') as f:
    json.dump(results, f, indent=2)
