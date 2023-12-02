from flask import Flask, request, jsonify
import librosa
import numpy as np
import joblib
import os
import tempfile
import xgboost as xgb

app = Flask(__name__)

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None, res_type='kaiser_fast')
    features = [
        librosa.feature.zero_crossing_rate(y),
        librosa.feature.spectral_centroid(y=y, sr=sr),
        librosa.feature.spectral_bandwidth(y=y, sr=sr),
        librosa.feature.spectral_rolloff(y=y, sr=sr),
        librosa.feature.mfcc(y=y, sr=sr),
    ]
    flat_features = np.concatenate([feature.mean(axis=1) for feature in features])
    selected_features = flat_features[:18]
    return selected_features

def predict_parkinsons(features):
    loaded_model = joblib.load('xgboost_model.joblib')
    features = features.reshape((1, -1))

    # Set use_label_encoder to False
    loaded_model._Booster.set_param('use_label_encoder', False)

    predictions = loaded_model.predict(features)
    return predictions[0]

@app.route('/')
def index():
    return jsonify('Sunday')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if the POST request has the file part
        if 'audio_path' not in request.files:
            return jsonify({'error': 'No file part'})

        audio_file = request.files['audio_path']

        # If the user does not select a file, the browser submits an empty part without a filename
        if audio_file.filename == '':
            return jsonify({'error': 'No selected file'})

        # Save the uploaded audio file to a temporary location
        temp_dir = tempfile.mkdtemp()
        temp_audio_path = os.path.join(temp_dir, 'temp_audio.wav')
        audio_file.save(temp_audio_path)

        # Extract features from the temporary audio file
        extracted_features = extract_features(temp_audio_path)

        # Predict Parkinson's using the extracted features
        prediction = predict_parkinsons(extracted_features)

        # Cleanup: Remove temporary audio file and directory
        os.remove(temp_audio_path)
        os.rmdir(temp_dir)

        # Conditional statements based on prediction
        if prediction == 1:
            result_message = 'Positive for Parkinson\'s'
            # Add any additional actions or messages for a positive prediction
        else:
            result_message = 'Negative for Parkinson\'s'
            # Add any additional actions or messages for a negative prediction

        return jsonify({'prediction': int(prediction), 'result_message': result_message})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
