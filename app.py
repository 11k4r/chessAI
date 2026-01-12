import os

from flask import Flask, render_template, request, jsonify
from config import Config 
from core.features import process_single_position 

import mimetypes
mimetypes.add_type('application/wasm', '.wasm')

app = Flask(__name__)
app.config.from_object(Config) # Load settings

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



@app.after_request
def add_header(response):
    response.headers['Cross-Origin-Embedder-Policy'] = 'require-corp'
    response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
    return response

    
if __name__ == '__main__':
    app.run(debug=True, port=5000)