## ML-Based Phishing Website Detection System

ML-Based Phishing Website Detection System is a professional-grade toolkit and web
application for detecting phishing websites using machine learning. It provides a
complete pipeline from feature extraction to model inference and a lightweight web
interface for real-time predictions and dataset submission.

Core capabilities

- Feature extraction: derive robust URL- and content-based features via
  `feature_extraction.py`.
- Preprocessing: consistent scaling using `feature_scaler.pkl`.
- Multiple model support: FFNN, TabNet, Wide & Deep (trained models included).
- Production-ready web UI: `app.py` exposes a Flask application for predictions.

Installation and quick start

1. Create a Python environment and install dependencies:

```bash
python -m venv venv
venv\Scripts\activate    # Windows
source venv/bin/activate  # macOS / Linux
pip install -r requirements.txt
```

2. Run the application:

```bash
python app.py
```

3. Open the web UI at `http://127.0.0.1:5000/` to test URLs and view performance.

Training

To retrain the default FFNN model from the dataset (optional):

```bash
python train_model.py
```

Use `--sample`, `--epochs`, and `--batch_size` to control training size and speed.

Notes

- Large trained model files are included for convenience. For collaborative
  workflows, consider using Git LFS to manage large binary artifacts.
- See `QUICK_START.md` for detailed deployment and evaluation instructions.

Contact & contributions

Contributions, issues, and feature requests are welcome. Please open a GitHub
issue or contact the repository owner for collaboration.
