from utils.Utils import load_model, load_vectorizer
from flask import Flask, request, render_template, jsonify
from utils.logger import logging

app = Flask(__name__)

@app.route('/')
def home():
    try:
       logging.info("Loading the home page")
       return render_template('index.html')
    except Exception as e :
        logging.error(f"Error in loading the home page: {e}")
        
        
@app.route('/newsclassification', methods=['POST'])
def classify_text():
    data = request.get_json()
    text = data.get("text", "")
    
    if not text.strip():
        return jsonify({"label": "error"})
    
    model = load_model()
    vectorizer = load_vectorizer()
    text_vector = vectorizer.transform([text])
    result = model.predict(text_vector)
    print(result)
    labels = {0: 'Fake News', 1: 'Real News'}
    
    return jsonify({"label": labels[result[0]]})

if __name__=="__main__":
    app.run(debug=True)