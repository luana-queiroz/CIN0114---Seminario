import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Multiply
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

RAND = 1337
np.random.seed(RAND)

train_files = [
    'Monday-WorkingHours.pcap_ISCX.csv',
    'Tuesday-WorkingHours.pcap_ISCX.csv',
    'Wednesday-workingHours.pcap_ISCX.csv',
    'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
    'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv'
]

test_files = [
    'Friday-WorkingHours-Morning.pcap_ISCX.csv',
    'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
    'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
]

df_train_full = pd.concat([pd.read_csv(f) for f in train_files])
df_test = pd.concat([pd.read_csv(f) for f in test_files])

df_train_full.columns = df_train_full.columns.str.strip()
df_test.columns = df_test.columns.str.strip()

df_train_full = df_train_full.replace([np.inf, -np.inf], np.nan).dropna(subset=['Flow Bytes/s'])
df_test = df_test.replace([np.inf, -np.inf], np.nan).dropna(subset=['Flow Bytes/s'])

logging.info("Data Preprocessing...")

df_train_full['Label'] = df_train_full['Label'].apply(lambda x: 0 if x == "BENIGN" else 1)
df_test['Label'] = df_test['Label'].apply(lambda x: 0 if x == "BENIGN" else 1)

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

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), make_column_selector(dtype_include=['number']))
    ])

X_train = preprocessor.fit_transform(X_train)
X_val = preprocessor.transform(X_val)
X_test = preprocessor.transform(X_test)

os.makedirs('./models', exist_ok=True)
with open('./models/preprocessor_v5.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

# ── Isolation Forest
logging.info("Training Isolation Forest to extract anomaly scores...")
iso = IsolationForest(n_estimators=200, random_state=RAND, n_jobs=-1)
iso.fit(X_train[y_train == 0])

with open('./models/model_iso_feature_extractor.pkl', 'wb') as f:
    pickle.dump(iso, f)

score_train = iso.decision_function(X_train).reshape(-1, 1)
score_val = iso.decision_function(X_val).reshape(-1, 1)
score_test = iso.decision_function(X_test).reshape(-1, 1)

# Concatena o score
X_train_aug = np.hstack((X_train, score_train))
X_val_aug = np.hstack((X_val, score_val))
X_test_aug = np.hstack((X_test, score_test))

logging.info(f"New X_train shape with IF score: {X_train_aug.shape}")

# ── DNN
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
        class_weight=class_weight_dict,
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
    class_weight=class_weight_dict,
    verbose=0
)

os.makedirs('./models', exist_ok=True)
model.save('./models/model_new_cic-ids-2017-v5.keras')

logging.info("Done. Evaluating Joint Pipeline...")

y_pred = (model.predict(X_test_aug) > 0.5).astype(int).flatten()

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average="binary")
precision = precision_score(y_test, y_pred, average="binary")
f1 = f1_score(y_test, y_pred, average="binary")

print("----------------------------------------------")
print("Pipeline Conjunta (DNN + Feature IF):")
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
with open('./models/results_pipeline_conjunta_cic-ids-2017-v5.json', 'w') as f:
    json.dump(results, f, indent=2)
