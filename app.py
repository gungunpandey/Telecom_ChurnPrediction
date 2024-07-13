from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.feature_extraction import DictVectorizer

# Load the trained model and DictVectorizer
model = pickle.load(open('model.pkl', 'rb'))
dv = pickle.load(open('dv.pkl', 'rb'))  # Load the DictVectorizer

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_churn():
    # Retrieve form data
    data = {
        'gender': request.form.get('gender'),
        'seniorcitizen': request.form.get('seniorcitizen'),
        'partner': request.form.get('partner'),
        'dependents': request.form.get('dependents'),
        'phoneservice': request.form.get('phoneservice'),
        'multiplelines': request.form.get('multiplelines'),
        'internetservice': request.form.get('internetservice'),
        'onlinesecurity': request.form.get('onlinesecurity'),
        'onlinebackup': request.form.get('onlinebackup'),
        'deviceprotection': request.form.get('deviceprotection'),
        'techsupport': request.form.get('techsupport'),
        'streamingtv': request.form.get('streamingtv'),
        'streamingmovies': request.form.get('streamingmovies'),
        'contract': request.form.get('contract'),
        'paperlessbilling': request.form.get('paperlessbilling'),
        'paymentmethod': request.form.get('paymentmethod'),
        'tenure': float(request.form.get('tenure')),
        'monthlycharges': float(request.form.get('monthlycharges')),
        'totalcharges': float(request.form.get('totalcharges'))
    }

    # Transform the input data using DictVectorizer
    data_encoded = dv.transform([data])

    # Prediction
    result = model.predict(data_encoded)

    return render_template('result.html', prediction=result[0])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081)
