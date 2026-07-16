from __future__ import annotations

import csv
import os
import sqlite3
from datetime import datetime, timezone
from typing import Optional, Any

import numpy as np
import pandas as pd
from flask import Flask, g, redirect, render_template, request, session, url_for
from feature_extraction import extract_features
from metrics_store import get_model_metrics_for_predict, get_performance_page_context, write_performance_metrics
from sklearn.model_selection import train_test_split
from werkzeug.security import check_password_hash, generate_password_hash

try:
    from tensorflow import keras
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    keras = None
    joblib = None

# NOTE: your template is located at `datasets/templates/index.html`
app = Flask(__name__, template_folder="datasets/templates")
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret-key-change-me")

CSV_PATH = os.path.join("datasets", "phishing_site_urls.csv")
MODEL_PATH = "trained_model.h5"
RF_MODEL_PATH = "trained_rf_model.pkl"
WD_MODEL_PATH = "trained_wd_model.h5"
TABNET_MODEL_PATH = "trained_tabnet_model.h5"
SCALER_PATH = "feature_scaler.pkl"
SUBMISSIONS_PATH = os.path.join("datasets", "user_submissions.csv")
DB_PATH = os.path.join("datasets", "app.db")

model: Optional[Any] = None  # keras.Model when available
rf_model: Optional[Any] = None  # RandomForestClassifier when available
wd_model: Optional[Any] = None  # keras.Model for Wide & Deep
tabnet_model: Optional[Any] = None  # keras.Model for TabNet-like
scaler: Optional[Any] = None  # StandardScaler when available
model_load_error: Optional[str] = None


def _train_model_from_csv(csv_path: str) -> Any:  # Returns keras.Model
    """Train a Feed Forward Neural Network from CSV data."""
    if keras is None:
        raise ImportError("TensorFlow/Keras is not available. Please install tensorflow.")
    
    # Load data
    df = pd.read_csv(csv_path)
    if "URL" not in df.columns or "Label" not in df.columns:
        raise ValueError("CSV must contain columns: URL, Label")

    df["Label"] = df["Label"].astype(str).str.strip().str.lower()
    df["Label"] = df["Label"].map({"good": 0, "bad": 1})
    df = df.dropna(subset=["URL", "Label"]).copy()
    df["Label"] = df["Label"].astype(int)

    # Balance the dataset - sample equal amounts of good and bad URLs
    good_urls = df[df["Label"] == 0]
    bad_urls = df[df["Label"] == 1]
    
    # Use balanced samples (minimum of the two classes, capped at 25k each)
    sample_size = min(len(good_urls), len(bad_urls), 25000)
    good_sample = good_urls.sample(n=sample_size, random_state=42)
    bad_sample = bad_urls.sample(n=sample_size, random_state=42)
    
    df_balanced = pd.concat([good_sample, bad_sample]).sample(frac=1, random_state=42)
    
    urls = df_balanced['URL'].astype(str).tolist()
    labels = df_balanced['Label'].astype(int).tolist()

    print(f"Training with balanced dataset: {len(urls)} URLs ({sample_size} good, {sample_size} bad)")

    X = np.array([extract_features(url) for url in urls])
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Normalize features
    from sklearn.preprocessing import StandardScaler
    local_scaler = StandardScaler()
    X_train_scaled = local_scaler.fit_transform(X_train)
    X_test_scaled = local_scaler.transform(X_test)
    
    # Save scaler + keep it available for in-process predictions
    global scaler
    scaler = local_scaler

    # Save scaler
    if joblib is not None:
        try:
            joblib.dump(local_scaler, SCALER_PATH)
        except Exception:
            pass
    
    # Build Feed Forward Neural Network (FFNN)
    m = keras.Sequential([
        # Input Layer
        keras.layers.Input(shape=(X_train_scaled.shape[1],)),
        
        # Hidden Layer 1
        keras.layers.Dense(256, activation='relu', name='hidden_1'),
        keras.layers.Dropout(0.3),
        
        # Hidden Layer 2
        keras.layers.Dense(128, activation='relu', name='hidden_2'),
        keras.layers.Dropout(0.3),
        
        # Hidden Layer 3
        keras.layers.Dense(64, activation='relu', name='hidden_3'),
        keras.layers.Dropout(0.2),
        
        # Hidden Layer 4
        keras.layers.Dense(32, activation='relu', name='hidden_4'),
        keras.layers.Dropout(0.2),
        
        # Output Layer
        keras.layers.Dense(1, activation='sigmoid', name='output')
    ])
    
    m.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Train with balanced class weights
    m.fit(
        X_train_scaled, y_train, 
        epochs=20, 
        batch_size=32, 
        validation_data=(X_test_scaled, y_test),
        verbose=0,
        class_weight={0: 1, 1: 1}  # Equal weight for both classes
    )
    # Train Random Forest Model
    from sklearn.ensemble import RandomForestClassifier
    print("Training Random Forest model...")
    local_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    local_rf.fit(X_train_scaled, y_train)
    
    # Train Wide & Deep Model
    print("Training Wide & Deep model...")
    wide_input = keras.layers.Input(shape=(X_train_scaled.shape[1],))
    deep_input = keras.layers.Input(shape=(X_train_scaled.shape[1],))
    deep = keras.layers.Dense(64, activation='relu')(deep_input)
    deep = keras.layers.Dropout(0.3)(deep)
    deep = keras.layers.Dense(32, activation='relu')(deep)
    deep = keras.layers.Dropout(0.3)(deep)
    merged = keras.layers.concatenate([wide_input, deep])
    output = keras.layers.Dense(1, activation='sigmoid')(merged)
    local_wd = keras.Model(inputs=[wide_input, deep_input], outputs=output)
    local_wd.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    local_wd.fit([X_train_scaled, X_train_scaled], y_train, epochs=2, batch_size=256, verbose=0)
    
    # Train TabNet-like Model
    print("Training TabNet model...")
    inputs = keras.layers.Input(shape=(X_train_scaled.shape[1],))
    attention = keras.layers.Dense(X_train_scaled.shape[1], activation='softmax')(inputs)
    gated = keras.layers.Multiply()([inputs, attention])
    x = keras.layers.Dense(64, activation='relu')(gated)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    output = keras.layers.Dense(1, activation='sigmoid')(x)
    local_tabnet = keras.Model(inputs=inputs, outputs=output)
    local_tabnet.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    local_tabnet.fit(X_train_scaled, y_train, epochs=2, batch_size=256, verbose=0)

    global rf_model, wd_model, tabnet_model
    rf_model = local_rf
    wd_model = local_wd
    tabnet_model = local_tabnet
    if joblib is not None:
        try:
            joblib.dump(local_rf, RF_MODEL_PATH)
        except Exception:
            pass
            
    try:
        local_wd.save(WD_MODEL_PATH)
        local_tabnet.save(TABNET_MODEL_PATH)
    except Exception:
        pass

    try:
        y_prob_ffnn = m.predict(X_test_scaled, verbose=0).flatten()
        y_prob_rf = local_rf.predict_proba(X_test_scaled)[:, 1].flatten()
        y_prob_wd = local_wd.predict([X_test_scaled, X_test_scaled], verbose=0).flatten()
        y_prob_tabnet = local_tabnet.predict(X_test_scaled, verbose=0).flatten()
        write_performance_metrics(
            y_test, 
            y_prob_ffnn, 
            y_prob_rf=y_prob_rf,
            y_prob_wd=y_prob_wd,
            y_prob_tabnet=y_prob_tabnet
        )
    except Exception:
        pass
    return m


def _load_or_train_model() -> Optional[Any]:  # Returns Optional[keras.Model]
    global model_load_error, scaler, rf_model, wd_model, tabnet_model

    if keras is None:
        model_load_error = "TensorFlow/Keras is not available. Please install tensorflow."
        return None

    # Fast path: load saved model
    if os.path.exists(MODEL_PATH):
        try:
            m = keras.models.load_model(MODEL_PATH)
            if joblib is not None:
                if os.path.exists(SCALER_PATH):
                    try:
                        scaler = joblib.load(SCALER_PATH)
                    except Exception:
                        model_load_error = "Model loaded but scaler failed to load. Predictions may be inaccurate."
                if os.path.exists(RF_MODEL_PATH):
                    try:
                        rf_model = joblib.load(RF_MODEL_PATH)
                    except Exception:
                        pass
            if os.path.exists(WD_MODEL_PATH):
                try:
                    wd_model = keras.models.load_model(WD_MODEL_PATH)
                except Exception:
                    pass
            if os.path.exists(TABNET_MODEL_PATH):
                try:
                    tabnet_model = keras.models.load_model(TABNET_MODEL_PATH)
                except Exception:
                    pass
            return m
        except Exception as e:
            model_load_error = f"Failed to load saved model: {e}"

    # Fallback: train from CSV (can be slow on large datasets)
    if not os.path.exists(CSV_PATH):
        model_load_error = f"Dataset not found: {CSV_PATH}"
        return None

    try:
        m = _train_model_from_csv(CSV_PATH)
        try:
            m.save(MODEL_PATH)
        except Exception:
            # Saving is optional; app can still run with in-memory model
            pass
        return m
    except Exception as e:
        model_load_error = f"Failed to train model: {e}"
        return None


# ---- DB helpers (SQLite) -----------------------------------------------------
def get_db() -> sqlite3.Connection:
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


def close_db(_: Optional[BaseException] = None) -> None:
    db = g.pop("db", None)
    if db is not None:
        db.close()


@app.teardown_appcontext
def _teardown_db(error: Optional[BaseException]) -> None:  # pragma: no cover
    close_db(error)


def init_db() -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    db = get_db()
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE, -- legacy column
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    # Backfill new columns if missing
    _ensure_column(db, "users", "first_name", "TEXT")
    _ensure_column(db, "users", "last_name", "TEXT")
    _ensure_column(db, "users", "email", "TEXT")
    _ensure_column(db, "users", "contact", "TEXT")
    try:
        db.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email)")
    except sqlite3.OperationalError:
        # If duplicates exist, index creation will fail; user can resolve manually.
        pass
    db.commit()


def _ensure_column(db: sqlite3.Connection, table: str, col: str, col_def: str) -> None:
    # Add a column if it does not exist (SQLite pragma-based check)
    existing = {r[1] for r in db.execute(f"PRAGMA table_info({table})").fetchall()}
    if col not in existing:
        db.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_def}")


def get_user_by_email(email: str) -> Optional[sqlite3.Row]:
    db = get_db()
    row = db.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    return row


def get_user_by_username(username: str) -> Optional[sqlite3.Row]:
    db = get_db()
    row = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    return row


def create_user(first_name: str, last_name: str, email: str, contact: str, password: str) -> None:
    db = get_db()
    username_value = email  # use email as username for login convenience
    db.execute(
        """
        INSERT INTO users (first_name, last_name, email, contact, username, password_hash, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            first_name,
            last_name,
            email,
            contact,
            username_value,
            generate_password_hash(password),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    db.commit()


with app.app_context():
    model = _load_or_train_model()
    init_db()

def _current_user():
    uid = session.get("user_id")
    email = session.get("email")
    display = session.get("display_name")
    if uid is None or email is None:
        return None
    return {"id": uid, "email": email, "display_name": display or email}


@app.route('/')
def home():
    user = _current_user()
    return render_template('home.html', user=user)


@app.route('/phishing')
def phishing():
    user = _current_user()
    if user is None:
        return redirect(url_for("login"))
    return render_template('phishing.html', user=user, results=None, url=None, error=None)


@app.route('/performance')
def performance():
    user = _current_user()
    ctx = get_performance_page_context()
    return render_template('performance.html', user=user, **ctx)


def _append_submission(url: str, prediction_text: str) -> None:
    """
    Append user submissions to a separate CSV.
    We intentionally do NOT write into the training dataset (unknown true label).
    """
    os.makedirs(os.path.dirname(SUBMISSIONS_PATH), exist_ok=True)
    is_new = not os.path.exists(SUBMISSIONS_PATH)
    with open(SUBMISSIONS_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["timestamp_utc", "url", "prediction"])
        w.writerow([datetime.now(timezone.utc).isoformat(), url, prediction_text])

def _detect_phishing_type(url: str, prob: float) -> str:
    """
    Determine the type of phishing based on URL characteristics.
    Returns: Phishing type string
    """
    url_lower = url.lower()

    # Obfuscation patterns (before generic login/signin heuristics so user@evil… is not mislabeled)
    if "@" in url or url.count(".") > 3:
        return "URL Obfuscation"

    # Common phishing patterns
    if any(keyword in url_lower for keyword in ["login", "signin", "verify", "account", "update"]):
        if any(keyword in url_lower for keyword in ["bank", "paypal", "amazon", "microsoft", "apple"]):
            return "Deceptive Phishing"
        return "Credential Harvesting"
    
    if any(keyword in url_lower for keyword in ['free', 'prize', 'winner', 'claim']):
        return "Baiting Phishing"
    
    if any(keyword in url_lower for keyword in ['support', 'help', 'service']):
        return "Tech Support Phishing"
    
    # Default phishing types based on probability
    if prob > 0.85:
        return "High-Risk Phishing"
    elif prob > 0.75:
        return "Deceptive Phishing"
    else:
        return "Suspicious Phishing"


def _predict_with_model(features_scaled: np.ndarray, model_type: str, url: str = "") -> tuple[str, float, str]:
    """
    Predict using the selected model type.
    Returns: (prediction_label, confidence, phishing_type)
    """
    if model_type == "random_forest":
        if rf_model is None:
            prob = np.random.uniform(0.3, 0.7)
        else:
            prob = rf_model.predict_proba(features_scaled)[0][1]
    elif model_type == "ffnn":
        if model is None:
            # Placeholder prediction if model not loaded
            prob = np.random.uniform(0.3, 0.7)
        else:
            prob = model.predict(features_scaled, verbose=0)[0][0]
    elif model_type == "wide_deep":
        if wd_model is None:
            prob = np.random.uniform(0.2, 0.6)
        else:
            prob = wd_model.predict([features_scaled, features_scaled], verbose=0)[0][0]
    elif model_type == "tabnet":
        if tabnet_model is None:
            prob = np.random.uniform(0.25, 0.65)
        else:
            prob = tabnet_model.predict(features_scaled, verbose=0)[0][0]
    else:
        # Default to FFNN
        if model is None:
            prob = np.random.uniform(0.3, 0.7)
        else:
            prob = model.predict(features_scaled, verbose=0)[0][0]
    
    # Determine prediction label with better thresholds
    # Use 0.5 as the boundary (binary classification)
    if prob > 0.6:
        label = "Phishing"
        phishing_type = _detect_phishing_type(url, prob)
    elif prob < 0.4:
        label = "Legitimate"
        phishing_type = "N/A"
    else:
        label = "Suspicious"
        # Still run URL heuristics so the UI can show a pattern name (grey-zone confidence)
        phishing_type = _detect_phishing_type(url, prob) + " (borderline)"

    return label, float(prob), phishing_type


@app.route('/predict', methods=['POST'])
def predict():
    user = _current_user()
    if user is None:
        return redirect(url_for("login"))

    url_input = (request.form.get("url") or "").strip()
    
    if not url_input:
        return render_template('phishing.html', 
                             results=None,
                             url=None, 
                             error="Please enter a URL.", 
                             user=user)
    
    # Basic URL validation
    if not (url_input.startswith("http://") or url_input.startswith("https://")):
        if "." not in url_input:
            return render_template('phishing.html', 
                                 results=None,
                                 url=url_input, 
                                 error="Please enter a valid URL (e.g., https://example.com).", 
                                 user=user)
        # Auto-add http:// if missing
        url_input = "http://" + url_input

    try:
        features = np.array([extract_features(url_input)])
        
        # Normalize features if scaler is available
        if scaler is not None:
            features_scaled = scaler.transform(features)
        else:
            features_scaled = features
        
        # Run all four models
        results = []
        model_types = ["random_forest", "ffnn", "wide_deep", "tabnet"]
        
        model_metrics = get_model_metrics_for_predict()
        for model_type in model_types:
            prediction, confidence, phishing_type = _predict_with_model(features_scaled, model_type, url_input)
            metrics = model_metrics[model_type]
            
            results.append({
                "model_name": metrics["name"],
                "prediction": prediction,
                "confidence": confidence,
                "phishing_type": phishing_type,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"]
            })
        
        # Save submission with majority vote
        try:
            predictions = [r["prediction"] for r in results]
            phishing_count = predictions.count("Phishing")
            suspicious_count = predictions.count("Suspicious")
            
            if phishing_count >= 2:
                majority_result = "Phishing"
            elif suspicious_count >= 2 or (phishing_count == 1 and suspicious_count >= 1):
                majority_result = "Suspicious"
            else:
                majority_result = "Legitimate"
            
            _append_submission(url_input, majority_result)
        except Exception:
            pass  # Submission saving is optional

        return render_template('phishing.html', 
                             results=results,
                             url=url_input, 
                             error=None, 
                             user=user)
    except Exception as e:
        return render_template('phishing.html', 
                             results=None,
                             url=url_input, 
                             error=f"Prediction failed: {e}", 
                             user=user)


@app.route("/signup", methods=["GET", "POST"])
def signup():
    user = _current_user()
    if user:
        return redirect(url_for("home"))

    error = None
    if request.method == "POST":
        first_name = (request.form.get("first_name") or "").strip()
        last_name = (request.form.get("last_name") or "").strip()
        email = (request.form.get("email") or "").strip().lower()
        contact = (request.form.get("contact") or "").strip()
        password = (request.form.get("password") or "").strip()
        if not first_name or not last_name or not email or not password:
            error = "All fields are required."
        elif "@" not in email:
            error = "Please enter a valid email."
        elif len(password) < 6:
            error = "Password must be at least 6 characters."
        elif get_user_by_email(email):
            error = "An account with this email already exists."
        else:
            try:
                create_user(first_name, last_name, email, contact, password)
                # Auto-login after signup
                user_row = get_user_by_email(email)
                session["user_id"] = user_row["id"]
                session["email"] = user_row["email"]
                session["display_name"] = (user_row["first_name"] or "").strip() or user_row["email"]
                return redirect(url_for("phishing"))
            except Exception as e:
                error = f"Could not create user: {e}"
    return render_template("signup.html", error=error)


@app.route("/login", methods=["GET", "POST"])
def login():
    user = _current_user()
    if user:
        return redirect(url_for("phishing"))

    error = None
    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        password = (request.form.get("password") or "").strip()
        row = get_user_by_email(email) if email else None
        if row and check_password_hash(row["password_hash"], password):
            session["user_id"] = row["id"]
            session["email"] = row["email"]
            session["display_name"] = (row["first_name"] or "").strip() or row["email"]
            return redirect(url_for("phishing"))
        else:
            error = "Invalid email or password."
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

if __name__ == '__main__':
    app.run(debug=True)
