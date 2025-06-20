# EEG Assistant 🧠🎛️

**EEG Assistant** is a web-based EEG analysis and classification toolkit built with **Flask**, **MNE-Python**, and **PyTorch**. It enables secure upload, preprocessing, feature extraction, and automatic detection of abnormal EEG patterns through a pretrained MLP model, along with interactive plotting and an optional chatbot interface.

---

## 🌟 Key Features

- **Web UI & API**  
  - Upload `.edf` EEG files via a Flask frontend.  
  - Asynchronous processing with progress tracking.  

- **Cloudflare R2 Storage**  
  - Secure upload/download of EEG files to Cloudflare R2.  
  - Configurable via `cloudflare_database.py`.

- **EEG Preprocessing & Visualization**  
  - Automatic band-pass/notch filtering, artifact removal, epoching.  
  - Interactive EEG window plots using **MNE-Python** (`raw.plot`).

- **Feature Extraction**  
  - Time‑domain: mean, skewness, kurtosis.  
  - Frequency‑domain: power spectral density (Welch).  
  - Wavelet features (**PyWavelets**).  
  - Catch22 feature set integration.  
  - Controlled by `selected_feature_names.txt`.

- **MLP Classification**  
  - Pretrained PyTorch MLP model (`mlp_best_model.pth`).  
  - Feature scaling with `scaler.pkl` (joblib).  
  - Predicts normal (0) vs abnormal (1) per window.

- **Chatbot Interface (Optional)**  
  - Query session results via DeepSeek or OpenAI GPT.  
  - Endpoint powered by `process_with_deepseek()` in `backend_flask.py`.

---

## 🔧 Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/osama-bassam/EEG-Assistant.git
   cd EEG-Assistant/EEG Assistant
   ```

2. **Create & activate a virtual environment**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

---

## ⚙️ Configuration

Create a `.env` or export the following environment variables before running:

```bash
export ACCESS_KEY="<YOUR_CLOUDFLARE_R2_ACCESS_KEY>"
export SECRET_KEY="<YOUR_CLOUDFLARE_R2_SECRET_KEY>"
export ENDPOINT_URL="<YOUR_CLOUDFLARE_R2_ENDPOINT>"
export DEEPSEEK_API_KEY="<YOUR_DEEPSEEK_OR_OPENAI_API_KEY>"
```

Alternatively, update the constants in `cloudflare_database.py` and `backend_flask.py`.

---

## 🚀 Usage

1. **Run the Flask app**  
   ```bash
   python backend_flask.py
   ```
   - Default: `http://127.0.0.1:5000/`

2. **Upload & Process**  
   - Navigate to the home page.  
   - Upload an `.edf` file; processing runs asynchronously.  

3. **View Results**  
   - Results appear dynamically on the page.  
   - Use the **Window** buttons to view 10 s EEG segments.  
   - Access `/how-it-works` and `/contact` for documentation and contact info.

4. **Production Deployment** (Heroku/Gunicorn)  
   ```bash
   gunicorn -w 4 backend_flask:app
   ```

---

## 📂 Repository Structure

```
EEG Assistant/
├── backend_flask.py         # Flask app & API endpoints
├── cloudflare_database.py   # R2 upload/download utilities
├── model_module.py          # EEG loading, preprocessing, feature extraction, plotting
├── mlp_best_model.pth       # Pretrained PyTorch MLP classifier
├── scaler.pkl               # Saved feature scaler (joblib)
├── selected_feature_names.txt
├── requirements.txt
├── runtime.txt              # Python version (Heroku)
├── MLP_model.ipynb          # Notebook: training & evaluation pipeline
├── data set.txt             # TUH Abnormal dataset reference & credentials
├── static/                  # CSS, JS, images
├── templates/               # Jinja2 HTML templates
├── uploads/, temp_edf/       # Runtime file storage
└── README.md                # (this file)
```

---

## 📚 Dataset

Data sourced from the **TUH EEG Abnormal** dataset:  
<https://isip.piconepress.com/projects/nedc/data/tuh_eeg/tuh_eeg_abnormal/>  
Credentials: see `data set.txt`.

---

## 🤝 Contributing

1. Fork the repo & create a branch.  
2. Follow **Conventional Commits** (`feat:`, `fix:`...).  
3. Ensure linting and tests (if any) pass.  
4. Submit a pull request.

---

## 📜 License

Distributed under the terms of `LICENSE.txt`.

---

## 📫 Contact

Osama Bassam — <osama.m.bassam@gmail.com>
