from flask import Flask, render_template, request, send_file, jsonify
import os
import tempfile
import torch
import joblib
import numpy as np
from model_module import load_raw_auto, preprocess, predict, plot_window, load_model
from io import BytesIO
import hashlib
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy.stats import skew, kurtosis
from scipy.signal import welch
import pywt
import openai
from pycatch22 import catch22_all
from threading import Thread
from cloudflare_database import upload_edf_to_r2, download_edf_from_r2
import requests


app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

SESSION = {}
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/how-it-works')
def how_it_works():
    return render_template('how-it-works.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')


#================================ChatBot====================================================================

# === Add this function to process queries with NLP (GPT or similar) ===


DEEPSEEK_API_KEY = 'sk-d66bec134a8f4177a13fffe11f0e473c'
DEEPSEEK_API_URL = 'https://api.deepseek.com/v1/chat/completions'
def process_with_deepseek(query, filehash):
    print(f"Fetching model output for filehash: {filehash}")

    # Fetch session data and get the abnormal count and total count
    session_data = SESSION.get(filehash)
    
    if not session_data:
        return "Session expired or invalid file.", 400

    abnormal = len(session_data['abnormal'])
    total = session_data['total']
    
    # Calculate the abnormal percentage
    abnormal_pct = 100 * abnormal / total if total > 0 else 0

    # Build the prompt for the chatbot (DeepSeek or OpenAI)
    prompt = f"""
    You are a helpful medical assistant. The user has undergone an EEG test.

    Their EEG shows {abnormal} abnormalities out of {total} segments ({abnormal_pct:.2f}% abnormality).

    Now answer the following user question in a friendly and informative tone:

    "{query}"
    """


    # Create the request payload for DeepSeek or OpenAI
    payload = {
        "model": "deepseek-chat",  # Or use "gpt-3.5-turbo" for OpenAI
        "messages": [
            {"role": "system", "content": "You are a friendly, helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 500
    }

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",  # Or OpenAI API key for GPT
        "Content-Type": "application/json"
    }

    # Send the request to DeepSeek or OpenAI API
    response = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers)

    if response.status_code == 200:
        # Return the dynamically generated response
        return response.json().get("choices")[0].get("message").get("content").strip()
    else:
        # Handle error case
        return f"Error: {response.status_code} - {response.text}"



# === Add the /query endpoint to handle user queries ===
@app.route('/query', methods=['POST'])
def handle_query():
    user_query = request.json.get('query', '')
    filehash = request.json.get('filehash', '')

    if not user_query or not filehash:
        return jsonify({"error": "Missing query or filehash"}), 400

    # Pass the filehash to process the result using the session data and DeepSeek/OpenAI API
    nlp_response = process_with_deepseek(user_query, filehash)

    return jsonify({"response": nlp_response})


def get_model_output(filehash):
    print(f"Fetching model output for filehash: {filehash}")  # Debug line to check filehash
    if filehash in SESSION:
        model_output = SESSION[filehash].get('model_output', None)
        print(f"Model Output Retrieved: {model_output}")  # Debug line to check if model_output is found
        return model_output
    return None




#====================================================End-OF-ChatBot==============================================================

def update_progress(filehash, progress):
    if filehash in SESSION:
        SESSION[filehash]['progress'] = progress
        print(f"Progress for {filehash}: {progress}%")

@app.route('/', methods=['GET', 'POST'])
def index():
    filehash = ''
    if request.method == 'POST':
        file = request.files['edfFile']
        if not file:
            return jsonify({'error': 'No file provided'}), 400

        byte_data = file.read()
        filehash = hashlib.md5(byte_data).hexdigest()
        SESSION[filehash] = {'progress': 0, 'complete': False}

        # Save temp file to upload to R2
        ext = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:

            tmp.write(byte_data)
            tmp_path = tmp.name

        # ✅ Upload to R2 before analysis
        upload_result = upload_edf_to_r2(tmp_path, update_progress=update_progress)

        if "already exists" in upload_result:
            # If the file exists, skip further upload steps and move to download
            print("[✅] File exists, moving to download process")
            tmp_path = download_edf_from_r2(filehash)  # Start download process
            thread = Thread(target=process_file, args=(filehash,))
            thread.start()
        else:
            # File upload successful, start processing
            thread = Thread(target=process_file, args=(filehash,))
            thread.start()

        return jsonify(filehash=filehash)

    return render_template("index.html")


def process_file(filehash):
    print(f"[🧠] Starting processing for: {filehash}")

    try:
        # Download the .edf file from R2
        tmp_path = download_edf_from_r2(filehash)
        raw = load_raw_auto(tmp_path)

        # Preprocess the raw EEG data
        X_eval, raw = preprocess(raw)

        # Load selected features from the file
        with open('selected_feature_names.txt') as f:
            selected_features = [line.strip() for line in f.readlines()]

        # Extract features for each window
        feats = []
        total_windows = len(X_eval)
        for i, w in enumerate(X_eval):
            features = single_window_feats(
                w.numpy(), selected_features,
                filehash=filehash, total_windows=total_windows, current_index=i
            )
            feats.append(features)

        # Convert the extracted features to a tensor
        X_feat = torch.tensor(np.stack(feats), dtype=torch.float32)

        # Load the scaler and model, then make predictions
        scaler = joblib.load('scaler.pkl')
        model = load_model('mlp_best_model.pth', input_dim=X_feat.shape[1])
        preds, confidences = predict(model, X_feat, scaler)

        # Identify abnormal indices
        abnormal_indices = (preds == 1).nonzero().squeeze().tolist()
        if isinstance(abnormal_indices, int):
            abnormal_indices = [abnormal_indices]

        # Store the results in the session (make sure it's set)
        SESSION[filehash].update({
            'raw': raw,
            'abnormal': abnormal_indices,
            'total': len(preds),
            'complete': True,
            'model_output': {
                'classification': preds.tolist(),
                'abnormal_indices': abnormal_indices,
                'confidence_scores': confidences.tolist()
            }
        })
        print(f"Model Output Stored: {SESSION[filehash]['model_output']}")

    except Exception as e:
        print(f"Error processing file: {e}")
        SESSION[filehash]['error'] = str(e)


    except Exception as e:
        print(f"Error processing file: {e}")
        SESSION[filehash]['error'] = str(e)

@app.route('/plot')
def plot():
    filehash = request.args.get('filehash')
    data = SESSION.get(filehash)
    if not data:
        return 'Session expired or invalid file.', 400

    channels_str = request.args.get('channels', '')
    channels = channels_str.split(',') if channels_str else data['raw'].ch_names[:18]

    try:
        index = int(request.args.get('index'))
        duration = 10  # Fixed value
    except:
        return 'Invalid parameters', 400

    idx_list = data['abnormal']
    if index >= len(idx_list):
        return 'Invalid index.', 400

    win_start = idx_list[index] * 10
    fig = plot_window(data['raw'], win_start, duration, channels)
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')


@app.route('/summary_chart')
@app.route('/summary_chart')
def summary_chart():
    filehash = request.args.get('filehash')
    session_data = SESSION.get(filehash)
    if not session_data:
        return 'Session expired', 400

    abnormal = len(session_data['abnormal'])
    total = session_data['total']
    normal = total - abnormal
    abnormal_pct = 100 * abnormal / total

    # ✅ Updated decision logic with 3 levels
    if abnormal_pct > 5:
        decision = "ABNORMAL"
    elif abnormal_pct >= 1:
        decision = "Anomaly Detected"
    else:
        decision = "NORMAL"

    fig, ax = plt.subplots(figsize=(12, 10), facecolor="#1c3a60")
    ax.set_facecolor("#1e3f67")

    ax.pie(
        [abnormal, normal],
        labels=['Abnormal', 'Normal'],
        autopct='%1.1f%%',
        colors=["#993535", "#2C9355"],
        startangle=90,
        textprops={'fontsize': 12, 'color': '#F0EFE9'}
    )

    ax.axis('equal')
    ax.text(
        0.02, 0.02,
        f"File decision: {decision}\nAbnormal: {abnormal} / {total} ({abnormal_pct:.2f}%)",
        transform=ax.transAxes,
        ha='left', va='bottom',
        fontsize=12, color='#F0EFE9'
    )

    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor="#1B385F")
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype='image/png')



@app.route('/progress')
def progress():
    filehash = request.args.get('filehash')
    session = SESSION.get(filehash, {})
    return jsonify(progress=session.get('progress', 0), complete=session.get('complete', False))

@app.route('/result')
def result():
    filehash = request.args.get('filehash')
    data = SESSION.get(filehash)
    if not data or not data.get('complete'):
        return "Still processing or session expired.", 400

    channels = data['raw'].ch_names
    max_index = len(data['abnormal']) - 1
    return render_template("partials/result.html", filehash=filehash, channels=channels, max_index=max_index)

def single_window_feats(window, selected_feature_names, sampling_rate=100, ch_idx_offset=0, filehash=None, total_windows=None, current_index=None):
    features = []
    for ch_idx, ch in enumerate(window):
        ch = np.nan_to_num(ch)
        if np.all(ch == ch[0]) or np.std(ch) < 1e-6:
            features.extend([0.0 for name in selected_feature_names if name.startswith(f"ch_{ch_idx}_")])
            continue
        if any("_time_" in f for f in selected_feature_names):
            if f"ch_{ch_idx}_time_mean" in selected_feature_names: features.append(ch.mean())
            if f"ch_{ch_idx}_time_std" in selected_feature_names: features.append(ch.std())
            if f"ch_{ch_idx}_time_max" in selected_feature_names: features.append(ch.max())
            if f"ch_{ch_idx}_time_min" in selected_feature_names: features.append(ch.min())
            if f"ch_{ch_idx}_time_skew" in selected_feature_names: features.append(skew(ch))
            if f"ch_{ch_idx}_time_kurtosis" in selected_feature_names: features.append(kurtosis(ch))
        if any("_psd_" in f for f in selected_feature_names):
            freqs, psd = welch(ch, fs=sampling_rate, nperseg=256)
            bands = {
                "delta": psd[(freqs >= 0.5) & (freqs < 4)].mean(),
                "theta": psd[(freqs >= 4) & (freqs < 8)].mean(),
                "alpha": psd[(freqs >= 8) & (freqs < 13)].mean(),
                "beta": psd[(freqs >= 13) & (freqs < 30)].mean(),
                "gamma": psd[(freqs >= 30) & (freqs < 40)].mean()
            }
            for band in bands:
                key = f"ch_{ch_idx}_psd_{band}"
                if key in selected_feature_names:
                    features.append(bands[band])
        if any("_hjorth_" in f for f in selected_feature_names):
            d1 = np.diff(ch); d2 = np.diff(d1)
            if f"ch_{ch_idx}_hjorth_var" in selected_feature_names: features.append(np.var(ch))
            if f"ch_{ch_idx}_hjorth_mob" in selected_feature_names: features.append(np.std(d1)/(np.std(ch)+1e-8))
            if f"ch_{ch_idx}_hjorth_comp" in selected_feature_names: features.append(np.std(d2)/(np.std(d1)+1e-8))
        if any("_wavelet_" in f for f in selected_feature_names):
            coeffs = pywt.wavedec(ch, 'db4', level=3)
            for i, c in enumerate(coeffs):
                key = f"ch_{ch_idx}_wavelet_cD{i}"
                if key in selected_feature_names:
                    features.append(np.sqrt(np.sum(c ** 2)))
        if any("_catch22_" in f for f in selected_feature_names):
            c22 = catch22_all(ch)["values"]
            for i, val in enumerate(c22):
                key = f"ch_{ch_idx}_catch22_{i}"
                if key in selected_feature_names:
                    features.append(np.nan_to_num(val))

    if filehash and total_windows and current_index is not None:
        progress_percent = int((current_index + 1) / total_windows * 100)
        SESSION[filehash]['progress'] = progress_percent

    return np.array(features, dtype=np.float32)

if __name__ == '__main__':
    app.run(debug=True, threaded=True, use_reloader=False)