import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, BatchNormalization, Multiply
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
import json
import os
import pickle
import optuna
from sklearn.utils.class_weight import compute_class_weight

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

RAND = 1337
np.random.seed(RAND)

train_files = [
    'Monday-WorkingHours.pcap_ISCX.csv',
    'Tuesday-WorkingHours.pcap_ISCX.csv',
    'Wednesday-workingHours.pcap_ISCX.csv',
    'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
    'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv'
]

# Test: Sexta completa
test_files = [
    'Friday-WorkingHours-Morning.pcap_ISCX.csv',          # Botnet
    'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv', # PortScan
    'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'      # DDoS
]

df_train_full = pd.concat([pd.read_csv(f) for f in train_files])
df_test = pd.concat([pd.read_csv(f) for f in test_files])

df_train_full.columns = df_train_full.columns.str.strip()
df_test.columns = df_test.columns.str.strip()

df_train_full = df_train_full.dropna(subset=['Flow Bytes/s'])
df_test = df_test.dropna(subset=['Flow Bytes/s'])

logging.info("Data Preprocessing...")

df_train_full['Label'] = df_train_full['Label'].apply(lambda x: 0 if x == "BENIGN" else 1)
df_test['Label'] = df_test['Label'].apply(lambda x: 0 if x == "BENIGN" else 1)
# Class 0 -> Benign
# Class 1 -> Attack

df_train_full = df_train_full.replace([np.inf, -np.inf], np.nan)
df_test = df_test.replace([np.inf, -np.inf], np.nan)
df_train_full = df_train_full.dropna(subset=['Flow Bytes/s'])
df_test = df_test.dropna(subset=['Flow Bytes/s'])

# Val: 20% do treino
df_train, df_val = train_test_split(df_train_full, test_size=0.2, random_state=RAND)

X_train = df_train.drop('Label', axis=1)
y_train = df_train['Label']
X_val = df_val.drop('Label', axis=1)
y_val = df_val['Label']
X_test = df_test.drop('Label', axis=1)
y_test = df_test['Label']

classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, weights))

logging.info(f"Class weights: {class_weight_dict}")

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(),
         make_column_selector(dtype_include=['number']))
    ])

X_train = preprocessor.fit_transform(X_train)
X_val = preprocessor.transform(X_val)
X_test = preprocessor.transform(X_test)

os.makedirs('./models', exist_ok=True)
with open('./models/preprocessor_v4.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

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
    epochs=100,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    class_weight=class_weight_dict,
    verbose=0
  )


  y_val_pred = (model.predict(X_val) > 0.5).astype(int).flatten()
  return f1_score(y_val, y_val_pred, average='binary')


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

logging.info("Best hyperparameters:")
for key, value in study.best_params.items():
  logging.info(f"{key}: {value}")

best_lr = study.best_params['lr']

checkpoint = ModelCheckpoint(
    './checkpoints/checkpoint.keras',
    monitor='val_loss',
    save_best_only=True
)

logging.info("Starting model training")

strategy = tf.distribute.MirroredStrategy()

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
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    class_weight=class_weight_dict,
    verbose=0
)
model.save('./models/model_new_cic-ids-2017-v4.keras')

logging.info("Done. Evaluating")

# Pipeline 
iso_final = IsolationForest(contamination=best_contamination, random_state=RAND, n_jobs=-1)
iso_final.fit(X_train[y_train == 0])

with open('./models/model_new_cic-ids-2017-v4_iso.pkl', 'wb') as f:
    pickle.dump(iso_final, f)

anomaly_pred = iso_final.predict(X_test)
X_suspicious = X_test[anomaly_pred == -1]

y_dnn_pred = (model.predict(X_suspicious) > 0.5).astype(int).flatten()

y_pred = np.zeros(len(X_test), dtype=int)
y_pred[anomaly_pred == -1] = y_dnn_pred

logging.info(f"Isolation Forest flagged {(anomaly_pred == -1).sum()} / {len(X_test)} samples as suspicious")

anomaly_pred_eval = iso_final.predict(X_test)
y_iso_only = (anomaly_pred_eval == -1).astype(int)

accuracy_iso = accuracy_score(y_test, y_iso_only)
recall_iso = recall_score(y_test, y_iso_only, average="binary")
precision_iso = precision_score(y_test, y_iso_only, average="binary")
f1_iso = f1_score(y_test, y_iso_only, average="binary")

print("----------------------------------------------")
print("Isolation Forest sozinho:")
print("accuracy:", "%.3f" % accuracy_iso)
print("recall:", "%.3f" % recall_iso)
print("precision:", "%.3f" % precision_iso)
print("f1score:", "%.3f" % f1_iso)

results_iso = {
    "accuracy": float(accuracy_iso),
    "recall": float(recall_iso),
    "precision": float(precision_iso),
    "f1": float(f1_iso)
}
with open('./models/results_iso_only_cic-ids-2017-v4.json', 'w') as f:
    json.dump(results_iso, f, indent=2)


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
with open('./models/results_model_new_cic-ids-2017-v4.json', 'w') as f:
    json.dump(results, f, indent=2)

