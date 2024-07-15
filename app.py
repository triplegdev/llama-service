from flask import Flask, request, jsonify
from transformers import LLaMAForSequenceClassification, LLaMATokenizer

app = Flask(__name__)

# Load pre-trained LLaMA model and tokenizer
model = LLaMAForSequenceClassification.from_pretrained('llama-small')
tokenizer = LLaMATokenizer.from_pretrained('llama-small')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json['text']
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
    predictions = outputs.last_hidden_state[:, 0, :]
    return jsonify({'predictions': predictions.tolist()})

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    texts = request.json['texts']
    inputs = tokenizer(texts, return_tensors='pt', padding=True)
    outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
    predictions = outputs.last_hidden_state[:, 0, :]
    return jsonify({'predictions': predictions.tolist()})

@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    return jsonify({'status': 'OK'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
