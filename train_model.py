import argparse
import os

import numpy as np
import pandas as pd
from feature_extraction import extract_features
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from tensorflow import keras
from tensorflow.keras import layers

from metrics_store import write_performance_metrics

csv_path = 'datasets/phishing_site_urls.csv'
model_path = 'trained_model.h5'
rf_model_path = 'trained_rf_model.pkl'
wd_model_path = 'trained_wd_model.h5'
tabnet_model_path = 'trained_tabnet_model.h5'
scaler_path = 'feature_scaler.pkl'

# Check if CSV exists
if not os.path.exists(csv_path):
    print(f"CSV file not found: {csv_path}")
    exit()

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a FFNN phishing-url detector.")
    p.add_argument("--max_rows", type=int, default=50_000, help="Max rows to load from the CSV (default: 50000). Use 0 for all.")
    p.add_argument("--sample", type=int, default=50_000, help="Random sample size after cleaning (default: 50000). Use 0 for no sampling.")
    p.add_argument("--epochs", type=int, default=5, help="Training epochs (default: 5).")
    p.add_argument("--batch_size", type=int, default=256, help="Batch size (default: 256).")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    return p.parse_args()


def build_wide_deep_model(input_dim):
    wide_input = layers.Input(shape=(input_dim,), name='wide_input')
    deep_input = layers.Input(shape=(input_dim,), name='deep_input')
    
    deep = layers.Dense(64, activation='relu')(deep_input)
    deep = layers.Dropout(0.3)(deep)
    deep = layers.Dense(32, activation='relu')(deep)
    deep = layers.Dropout(0.3)(deep)
    
    merged = layers.concatenate([wide_input, deep])
    output = layers.Dense(1, activation='sigmoid')(merged)
    
    model = keras.Model(inputs=[wide_input, deep_input], outputs=output)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def build_tabnet_like_model(input_dim):
    inputs = layers.Input(shape=(input_dim,), name='tabnet_input')
    
    attention_weights = layers.Dense(input_dim, activation='softmax', name='attention')(inputs)
    gated_features = layers.Multiply()([inputs, attention_weights])
    
    x = layers.Dense(64, activation='relu')(gated_features)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation='relu')(x)
    
    output = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs=inputs, outputs=output)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def main():
    args = _parse_args()
    print("Loading dataset...")
    # Load CSV
    nrows = None if args.max_rows == 0 else args.max_rows
    df = pd.read_csv(csv_path, nrows=nrows, usecols=["URL", "Label"])

    # Clean & convert labels
    df['Label'] = df['Label'].str.strip().str.lower()
    df['Label'] = df['Label'].map({'good': 0, 'bad': 1})
    df = df.dropna(subset=["URL", "Label"]).copy()

    if args.sample and args.sample > 0 and len(df) > args.sample:
        df = df.sample(n=args.sample, random_state=args.seed).reset_index(drop=True)
        print(f"Using random sample of {len(df)} rows")

    urls = df['URL'].astype(str).tolist()
    labels = df['Label'].astype(int).tolist()
    print(f"Extracting features from {len(urls)} URLs...")
    
    # Extract features with error handling for invalid URLs
    X_list = []
    y_list = []
    skipped = 0
    for url, label in zip(urls, labels):
        try:
            features = extract_features(url)
            X_list.append(features)
            y_list.append(label)
        except Exception as e:
            skipped += 1
            if skipped <= 5:  # Show first few errors
                print(f"Warning: Skipping invalid URL: {url[:50]}... ({e})")
    
    if skipped > 0:
        print(f"Skipped {skipped} invalid URLs out of {len(urls)} total")
    
    X = np.array(X_list)
    y = np.array(y_list)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )

    # Normalize features for neural network
    print("Normalizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler for inference
    import joblib
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved as {scaler_path}")

    # Build Feed Forward Neural Network
    print("Building Feed Forward Neural Network...")
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print("Model architecture:")
    model.summary()

    # Train model
    print("\nTraining model...")
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
        )
    ]

    model.fit(
        X_train_scaled,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.1,
        verbose=1,
        callbacks=callbacks,
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    y_pred_proba = model.predict(X_test_scaled, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy on test set: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Safe', 'Phishing']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=args.seed)
    rf_model.fit(X_train_scaled, y_train)
    y_pred_proba_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

    print("\nTraining Wide & Deep Model...")
    wd_model = build_wide_deep_model(X_train_scaled.shape[1])
    wd_model.fit(
        [X_train_scaled, X_train_scaled], y_train,
        epochs=args.epochs, batch_size=args.batch_size, validation_split=0.1, verbose=1, callbacks=callbacks
    )
    y_pred_proba_wd = wd_model.predict([X_test_scaled, X_test_scaled], verbose=0)

    print("\nTraining TabNet-like Model...")
    tabnet_model = build_tabnet_like_model(X_train_scaled.shape[1])
    tabnet_model.fit(
        X_train_scaled, y_train,
        epochs=args.epochs, batch_size=args.batch_size, validation_split=0.1, verbose=1, callbacks=callbacks
    )
    y_pred_proba_tabnet = tabnet_model.predict(X_test_scaled, verbose=0)

    try:
        write_performance_metrics(
            y_test, 
            y_pred_proba.flatten(), 
            y_prob_rf=y_pred_proba_rf.flatten(),
            y_prob_wd=y_pred_proba_wd.flatten(),
            y_prob_tabnet=y_pred_proba_tabnet.flatten()
        )
    except Exception as e:
        print(f"Warning: could not save performance metrics JSON: {e}")

    # Save models
    model.save(model_path)
    joblib.dump(rf_model, rf_model_path)
    wd_model.save(wd_model_path)
    tabnet_model.save(tabnet_model_path)
    print(f"\nFFNN Model saved as {model_path}")
    print(f"Random Forest Model saved as {rf_model_path}")
    print(f"Wide & Deep Model saved as {wd_model_path}")
    print(f"TabNet Model saved as {tabnet_model_path}")
    print(f"Scaler saved as {scaler_path}")


if __name__ == "__main__":
    main()
