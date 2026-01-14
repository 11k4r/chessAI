import os
from flask import Flask, render_template, request, jsonify
from config import Config 
from core.features import process_single_position 

import mimetypes
mimetypes.add_type('application/wasm', '.wasm')

app = Flask(__name__)
app.config.from_object(Config) # Load settings

import asyncio
import sys
# 1. Windows Fix
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

@app.route('/')
def index():
    return render_template('index.html', 
                           site_name=app.config['SITE_NAME'], # Accessed via app.config
                           tagline="Push Chess Forward",
                           sub_tagline="Go beyond the evaluation bar and decode your chess DNA. Use AI to measure key metrics such as harmony, mobility, pawn structure, and time management. Seamlessly sync with Chess.com and Lichess to transform your game history into a comprehensive player profile.")

@app.route('/analyze-game')
def analyze_game():
    return render_template('analyze.html', 
                           site_name=app.config['SITE_NAME'],
                           tc_styles=app.config['TIME_CONTROL_STYLES'])

@app.route('/manual')
def manual():
    return render_template('manual.html', site_name=app.config['SITE_NAME'])



@app.route('/api/analyze-batch', methods=['POST'])
def analyze_batch():
    """
    Receives a batch of positions (FENs + Client Dynamic Data),
    runs the Static Engine fusion, and returns calculated features.
    """
    data = request.get_json()

    
    results = []
    
    # Ensure this path is defined in your Config (see step 2)
    static_engine_path = app.config.get('STATIC_ENGINE_PATH')
    
    if not static_engine_path or not os.path.exists(static_engine_path):
        print("Error: Static engine path not configured or file missing.")
        return jsonify({"error": "Server configuration error: Engine not found"}), 500
    for i in range(len(data['positions'])):      
        try:
            # process_single_position is already imported from core.features
            features = process_single_position(static_engine_path, i, data)
            
            # Structure the response for the frontend
            results.append(features)
            
        except Exception as e:
            print(f"Error processing FEN {i}: {e}")
            print(data)
            results.append({"move_index": i, "error": str(e)})

    return jsonify({"data": results})

@app.after_request
def add_header(response):
    response.headers['Cross-Origin-Embedder-Policy'] = 'require-corp'
    response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
    return response

    
if __name__ == '__main__':
    app.run(debug=True, port=5000)