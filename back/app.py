from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)

# Habilita CORS para todas as rotas
CORS(app)

# Carregar modelo
modelo = joblib.load('model.joblib')

# Colunas usadas no treino
COLUNAS = [
    "Unnamed: 0": 1,
    "Battery capacity (mAh)": 5000,
    "Screen size (inches)": 6.5,
    "Touchscreen": 1,
    "Resolution x": 1080,
    "Resolution y": 2400,
    "Processor": 8,
    "RAM (MB)": 8192,
    "Internal storage (GB)": 128,
    "Rear camera": 48,
    "Front camera": 16,
    "Wi-Fi": 1,
    "Bluetooth": 1,
    "GPS": 1,
    "Number of SIMs": 2,
    "3G": 1,
    "4G/ LTE": 1
]

@app.route('/')
def home():
    return 'API de previsão de preço está rodando'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Criar DataFrame com ordem correta
    X = pd.DataFrame([data], columns=COLUNAS)

    # Garantir tipos numéricos
    X = X.astype(float)

    preco = modelo.predict(X)[0]

    return jsonify({
        'preco_previsto': round(float(preco), 2)
    })

@app.route('/health')
def health():
    return {'status': 'ok'}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
