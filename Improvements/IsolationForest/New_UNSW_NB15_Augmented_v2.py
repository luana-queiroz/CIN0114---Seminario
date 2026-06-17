import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, BatchNormalization, Multiply
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
import json
import pickle
import optuna
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

RAND = 1337
np.random.seed(RAND)

DATA_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "./"

data = pd.read_csv(os.path.join(DATA_DIR, './Data.csv'))
labels = pd.read_csv(os.path.join(DATA_DIR, './Label.csv'))
df = pd.concat([data, labels], axis=1)

logging.info("Data Preprocessing...")

labels_dict = {
    "Benign": 0,
    "Analysis": 1,
    "Backdoor": 2,
    "DoS": 3,
    "Exploits": 4,
    "Fuzzers": 5,
    "Generic": 6,
    "Reconnaissance": 7,
    "Shellcode": 8,
    "Worms": 9
}

test_attacks = ['Analysis', 'Backdoor', 'Shellcode', 'Worms']

df_benign = df[df['Label'] == labels_dict['Benign']]

test_labels = [labels_dict[attack] for attack in test_attacks if attack in labels_dict]
df_test_attacks = df[df['Label'].isin(test_labels)]

df_train_attacks = df[(df['Label'] != labels_dict['Benign']) & (~df['Label'].isin(test_labels))]

b_train, b_temp = train_test_split(df_benign, test_size=0.30, random_state=RAND)
b_val, b_test = train_test_split(b_temp, test_size=0.50, random_state=RAND)
a_train, a_val = train_test_split(df_train_attacks, test_size=0.20, random_state=RAND)

df_train = pd.concat([b_train, a_train]).sample(frac=1, random_state=RAND).reset_index(drop=True)
df_val = pd.concat([b_val, a_val]).sample(frac=1, random_state=RAND).reset_index(drop=True)
df_test = pd.concat([b_test, df_test_attacks]).sample(frac=1, random_state=RAND).reset_index(drop=True)

df_train['Label'] = df_train['Label'].apply(lambda x: min(x, 1))
df_val['Label']   = df_val['Label'].apply(lambda x: min(x, 1))
df_test['Label']  = df_test['Label'].apply(lambda x: min(x, 1))

print("--- Tamanhos ---")
print(f"Treino: {len(df_train)} | Validação: {len(df_val)} | Teste: {len(df_test)}")
print("\n--- Ataques no Teste ---")
print(df_test['Label'].value_counts())

X_train = df_train.drop('Label', axis=1)
y_train = df_train['Label']
X_val = df_val.drop('Label', axis=1)
y_val = df_val['Label']
X_test = df_test.drop('Label', axis=1)
y_test = df_test['Label']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), make_column_selector(dtype_include=['number']))
    ])

X_train = preprocessor.fit_transform(X_train)
X_val = preprocessor.transform(X_val)
X_test = preprocessor.transform(X_test)

os.makedirs('./models', exist_ok=True)
with open('./models/preprocessor_v1.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

logging.info(f"X_train shape originally: {X_train.shape}")

# ── Isolation Forest
logging.info("Training Isolation Forest to extract anomaly scores...")

iso = IsolationForest(n_estimators=200, random_state=RAND, n_jobs=-1)
iso.fit(X_train[y_train.values == 0]) 

with open('./models/model_iso_feature_extractor_v1.pkl', 'wb') as f:
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

classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, weights))
logging.info(f"Class weights: {class_weight_dict}")

logging.info("Starting hyperparameter optimization for DNN")
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

        # Camada de atenção
        attention = Dense(512, activation='softmax')(x)
        x = Multiply()([x, attention])

        output = Dense(1, activation='sigmoid')(x)

        model = tf.keras.Model(inputs=inputs, outputs=output)

        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['accuracy'])

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    model.fit(
        X_train_aug, y_train,
        batch_size=BATCH_SIZE,
        epochs=100, # Optuna avalia em 100 épocas
        validation_data=(X_val_aug, y_val),
        callbacks=[early_stop],
        class_weight=class_weight_dict,
        verbose=0
    )

    y_val_pred = (model.predict(X_val_aug) > 0.5).astype(int).flatten()
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

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

model.fit(
    X_train_aug, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val_aug, y_val),
    callbacks=[early_stop, checkpoint],
    class_weight=class_weight_dict,
    verbose=1
)

model.save('./models/model_new_cic_unsw-nb15-v1.keras')

logging.info("Done. Evaluating Joint Pipeline...")

# Avaliação Final (DNN processa as features originais + escore do IF)
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

with open('./models/results_v2.json', 'w') as f:
    json.dump(results, f, indent=2)
