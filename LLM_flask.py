from flask import Flask, request, jsonify, render_template
from LLM_code import *

app = Flask(__name__, static_folder='static', static_url_path='/static')

@app.route('/')
def index():
    return render_template('LLM_html-v1.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('query', '')
    success, response, history = agent_execute_with_retry(query)
    return jsonify({'response': response})

if __name__ == '__main__':
    os.makedirs('static/images/', exist_ok=True)
    os.makedirs('static/audio/', exist_ok=True)
    app.run(port=5000)