import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Multiply
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import optuna
import tensorflow as tf
from keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import json
import os
import pickle
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

RAND = 1337
np.random.seed(RAND)
tf.random.set_seed(RAND)

DATA_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else "./"

logging.info("Loading Data...")
train = pd.read_csv(os.path.join(DATA_DIR, 'Training.csv'), header=None)
test = pd.read_csv(os.path.join(DATA_DIR, 'Testing.csv'), header=None)

X_train_df = train.iloc[:, 1:42]
y_train_df = train.iloc[:, 0]
X_test_df = test.iloc[:, 1:42]
y_test_df = test.iloc[:, 0]

X_train_df, X_val_df, y_train_df, y_val_df = train_test_split(
    X_train_df, y_train_df, test_size=0.2, random_state=RAND
)

logging.info("Data Preprocessing (Normalizer)...")
scaler = Normalizer().fit(X_train_df)

X_train = np.array(scaler.transform(X_train_df))
X_val = np.array(scaler.transform(X_val_df))
X_test = np.array(scaler.transform(X_test_df))

y_train = np.array(y_train_df)
y_val = np.array(y_val_df)
y_test = np.array(y_test_df)

# ── Isolation Forest: Extração de Features 
logging.info("Training Isolation Forest to extract anomaly scores...")
iso = IsolationForest(n_estimators=200, random_state=RAND, n_jobs=-1)
iso.fit(X_train[y_train == 0])

os.makedirs(os.path.join(DATA_DIR, 'models'), exist_ok=True)
with open(os.path.join(DATA_DIR, 'models', 'model_iso_kdd_feature_extractor.pkl'), 'wb') as f:
    pickle.dump(iso, f)

score_train = iso.decision_function(X_train).reshape(-1, 1)
score_val = iso.decision_function(X_val).reshape(-1, 1)
score_test = iso.decision_function(X_test).reshape(-1, 1)

# Concatena o score 
X_train_aug = np.hstack((X_train, score_train))
X_val_aug = np.hstack((X_val, score_val))
X_test_aug = np.hstack((X_test, score_test))

logging.info(f"New X_train shape with IF score: {X_train_aug.shape}")

# DNN
BATCH_SIZE = 1024
EPOCHS = 1000

strategy = tf.distribute.MirroredStrategy()

def objective(trial):
    with strategy.scope():
        inputs = tf.keras.Input(shape=(X_train_aug.shape[1],))
        
        x = Dense(1024, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = Dense(768, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        attention = Dense(512, activation='softmax')(x)
        x = Multiply()([x, attention])
        
        output = Dense(1, activation='sigmoid')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=output)
        
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['accuracy'])
        
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    model.fit(
        X_train_aug, y_train,
        batch_size=BATCH_SIZE,
        epochs=100,
        validation_data=(X_val_aug, y_val),
        callbacks=[early_stop],
        verbose=0
    )
    
    y_val_pred = (model.predict(X_val_aug) > 0.5).astype(int).flatten()
    return f1_score(y_val, y_val_pred, average='binary')

logging.info("Starting DNN hyperparameter optimization")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)
best_lr = study.best_params['lr']

logging.info(f"Best LR: {best_lr}")
logging.info("Starting final model training")

with strategy.scope():
    inputs = tf.keras.Input(shape=(X_train_aug.shape[1],))
    
    x = Dense(1024, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Dense(768, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    attention = Dense(512, activation='softmax')(x)
    x = Multiply()([x, attention])
    
    output = Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=best_lr), metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model.fit(
    X_train_aug, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val_aug, y_val),
    callbacks=[early_stop],
    verbose=1
)

model.save(os.path.join(DATA_DIR, 'models', 'model_new_kdd_v4.keras'))
logging.info("Done. Evaluating Joint Pipeline and Isolation Forest...")

# --- Avaliação: Isolation Forest Sozinho ---
y_iso_pred = iso.predict(X_test)
y_iso_only = (y_iso_pred == -1).astype(int)

accuracy_iso = accuracy_score(y_test, y_iso_only)
recall_iso = recall_score(y_test, y_iso_only, average="binary")
precision_iso = precision_score(y_test, y_iso_only, average="binary")
f1_iso = f1_score(y_test, y_iso_only, average="binary")

print("----------------------------------------------")
print("Isolation Forest Sozinho (Contamination=Auto):")
print(f"accuracy: {accuracy_iso:.3f}")
print(f"recall: {recall_iso:.3f}")
print(f"precision: {precision_iso:.3f}")
print(f"f1score: {f1_iso:.3f}")

results_iso = {
    "accuracy": float(accuracy_iso),
    "recall": float(recall_iso),
    "precision": float(precision_iso),
    "f1": float(f1_iso)
}
with open(os.path.join(DATA_DIR, 'models', 'results_iso_only_kdd_v4.json'), 'w') as f:
    json.dump(results_iso, f, indent=2)

# --- Avaliação: Pipeline Conjunta
y_pred = (model.predict(X_test_aug) > 0.5).astype(int).flatten()

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average="binary")
precision = precision_score(y_test, y_pred, average="binary")
f1 = f1_score(y_test, y_pred, average="binary")

print("----------------------------------------------")
print("Pipeline Conjunta (DNN + Feature IF) - KDD:")
print(f"accuracy: {accuracy:.3f}")
print(f"recall: {recall:.3f}")
print(f"precision: {precision:.3f}")
print(f"f1score: {f1:.3f}")

results = {
    "accuracy": float(accuracy),
    "recall": float(recall),
    "precision": float(precision),
    "f1": float(f1)
}
with open(os.path.join(DATA_DIR, 'models', 'results_model_new_kdd_v4.json'), 'w') as f:
    json.dump(results, f, indent=2)
