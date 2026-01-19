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
    "Unnamed: 0",
    "Battery capacity (mAh)",
    "Screen size (inches)",
    "Touchscreen",
    "Resolution x",
    "Resolution y",
    "Processor",
    "RAM (MB)",
    "Internal storage (GB)",
    "Rear camera",
    "Front camera",
    "Wi-Fi",
    "Bluetooth",
    "GPS",
    "Number of SIMs",
    "3G",
    "4G/ LTE"
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
