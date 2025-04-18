from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load('wine_quality_model.pkl')
scaler = joblib.load('wine_scaler.pkl')
encoder = joblib.load('label_encoder.pkl')

feature_names =['fixed acidity', 'volatile acidity', 'citric acid', 
                'residual sugar', 'chlorides', 'free sulfur dioxide',
                'total sulfur dioxide','density', 'pH', 'sulphates',
                'alcohol']

@app.route('/')
def home():
    return render_template('index.html', feature_names=feature_names)
                           
@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_features = [float(request.form[f'feature{i}']) for i in range(1, 12)]
        input_array = np.array(input_features).reshape(1, -1)
        scaled_input = scaler.transform(input_array)
        predicted_number = model.predict(scaled_input)[0]
        predicted_letter = encoder.inverse_transform([predicted_number])[0]

        return render_template('index.html', prediction=f'Predicted Quality: {predicted_letter}', feature_names=feature_names)
    
    except Exception as e:
        return render_template('index.html', prediction=f'Error: {e}')

if __name__ == '__main__':
    app.run(debug=True)