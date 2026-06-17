import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, BatchNormalization, Multiply
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
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

DATA_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else "./"

train = pd.read_csv(os.path.join(DATA_DIR, 'UNSW_NB15_training-set.csv'))
test = pd.read_csv(os.path.join(DATA_DIR, 'UNSW_NB15_testing-set.csv'))

logging.info("Data Preprocessing...")

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

os.makedirs(os.path.join(DATA_DIR, 'models'), exist_ok=True)
with open(os.path.join(DATA_DIR, 'models', 'preprocessor_v3.pkl'), 'wb') as f:
    pickle.dump(preprocessor, f)

X_train = preprocessor.fit_transform(X_train_df).toarray()
X_val = preprocessor.transform(X_val_df).toarray()
X_test = preprocessor.transform(X_test_df).toarray()

y_train = np.array(y_train_df)
y_val = np.array(y_val_df)
y_test = np.array(y_test_df)

logging.info(f"X_train shape originally: {X_train.shape}")

# ── Isolation Forest
logging.info("Training Isolation Forest to extract anomaly scores...")

iso = IsolationForest(n_estimators=200, random_state=RAND, n_jobs=-1)
iso.fit(X_train[y_train == 0])

with open(os.path.join(DATA_DIR, 'models', 'model_unsw_iso_feature_extractor.pkl'), 'wb') as f:
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
        epochs=100, # Optuna em menos épocas para velocidade
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
    callbacks=[early_stop],
    class_weight=class_weight_dict,
    verbose=0
)

model.save(os.path.join(DATA_DIR, 'models', 'model_new_unsw_nb15_v3.keras'))

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

with open(os.path.join(DATA_DIR, 'models', 'results_pipeline_conjunta_unsw_nb15-v3.json'), 'w') as f:
    json.dump(results, f, indent=2)
