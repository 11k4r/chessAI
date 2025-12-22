from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import os
import sys
import requests
import random
import asyncio
import json
import threading
import queue
import io
import chess.pgn
import stat

# Ensure the current directory is in the python path
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.append(base_dir)

# Try to handle Windows Loop Policy for Stockfish
if sys.platform == 'win32':
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
        pass

# --- IMPORTS ---
from chessAI.analyzer import ChessAnalyzer
from chessAI.core.engine import EngineHandler

app = Flask(__name__)

# --- HELPER: Get Engine Path ---
def get_engine_path():
    """Returns the correct engine binary path based on the OS."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    if sys.platform == 'win32':
        # Windows
        path = os.path.join(base_dir, "engines", "stockfish.exe")
    else:
        # Linux (Render) / Mac
        path = os.path.join(base_dir, "engines", "stockfish")
        
        # Ensure it is executable (Linux only)
        if os.path.exists(path):
            try:
                # Force 755 permissions (rwxr-xr-x)
                # This allows Read/Write/Execute for Owner, and Read/Execute for everyone else.
                os.chmod(path, 0o755)
            except Exception as e:
                print(f"⚠️ Warning: Failed to set permissions for engine: {e}")
            
    return path

# --- CONFIGURATION ---
SITE_CONFIG = {
    "site_name": "ChessAI",
    "tagline": "Push Chess Forward.",
    "sub_tagline": "Unlock 20+ advanced metrics including calculating intuition, pressure handling, and strategic resourcefulness. The next evolution of engine analysis is here.",
    "nav_links": [
        {"name": "Home", "url": "/", "icon": "fa-house", "desc": "Return to the main dashboard."},
        {"name": "Analyze", "url": "/analyze-game", "icon": "fa-magnifying-glass-chart", "desc": "Find and analyze your recent games."},
        {"name": "Guide", "url": "/guide", "icon": "fa-book-open", "desc": "Learn how our metrics like 'Intuition' and 'Harmony' work."},
        {"name": "About", "url": "/about", "icon": "fa-circle-info", "desc": "Meet the engine and the logic behind the metrics."}
    ]
}

@app.route('/')
def home():
    return render_template('index.html', **SITE_CONFIG)

@app.route('/analyze-game')
def games_page():
    return render_template('games.html', **SITE_CONFIG)

@app.route('/analyzer')
def analyzer_page():
    return render_template('analyze.html', **SITE_CONFIG)

@app.route('/guide')
def guide_page():
    return render_template('guide.html', **SITE_CONFIG)

@app.route('/about')
def about_page():
    return render_template('about.html', **SITE_CONFIG)

# --- GAMES SEARCH API ---

@app.route('/api/search-player', methods=['POST'])
def search_player():
    data = request.json
    username = data.get('username')
    
    if not username:
        return jsonify({"error": "Username required"}), 400
        
    try:
        # Fetch Archives
        url = f"https://api.chess.com/pub/player/{username}/games/archives"
        resp = requests.get(url, headers={"User-Agent": "ChessAI-Bot"})
        
        if resp.status_code == 404:
             return jsonify({"error": "User not found"}), 404
             
        archives = resp.json().get("archives", [])
        
        # Return reversed (newest first)
        return jsonify({"archives": list(reversed(archives))})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/fetch-archive', methods=['POST'])
def fetch_archive():
    data = request.json
    url = data.get('url')
    
    if not url: return jsonify({"error": "URL required"}), 400
    
    try:
        resp = requests.get(url, headers={"User-Agent": "ChessAI-Bot"})
        games = resp.json().get("games", [])
        return jsonify({"games": games})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- ANALYSIS API ---

@app.route('/api/analyze-position', methods=['POST'])
def analyze_position():
    data = request.json
    fen = data.get('fen')
    
    if not fen:
        return jsonify({"error": "No FEN provided"}), 400

    try:
        engine_path = get_engine_path()
        if not os.path.exists(engine_path):
             return jsonify({"error": f"Stockfish binary not found at {engine_path}"}), 500

        analyzer = ChessAnalyzer(engine_path=engine_path)
        with EngineHandler(engine_path) as engine:
            metrics = analyzer.analyze_position(fen, engine_handler=engine)
            
        return jsonify({"status": "success", "metrics": metrics})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/analyze-pgn', methods=['POST'])
def analyze_pgn_stream():
    """
    Streams analysis for a SPECIFIC PGN provided by the client.
    """
    pgn_text = request.json.get('pgn')
    if not pgn_text:
        return jsonify({"error": "No PGN provided"}), 400

    result_queue = queue.Queue()
    stop_event = threading.Event()
    
    def analysis_task():
        try:
            # 1. Parse Initial Headers
            pgn_io = io.StringIO(pgn_text)
            parsed_game = chess.pgn.read_game(pgn_io)
            if not parsed_game: raise Exception("Invalid PGN")
            
            total_moves = sum(1 for _ in parsed_game.mainline_moves())
            
            result_queue.put({
                "type": "start",
                "total_moves": total_moves,
                "pgn": pgn_text,
                "headers": dict(parsed_game.headers)
            })

            # 2. Run Analysis
            engine_path = get_engine_path()
            if not os.path.exists(engine_path): 
                raise Exception(f"Stockfish not found at {engine_path}")

            analyzer = ChessAnalyzer(engine_path=engine_path)
            
            class AnalysisCancelled(Exception): pass

            def on_move_analyzed():
                if stop_event.is_set(): raise AnalysisCancelled()
                result_queue.put({"type": "progress"})

            try:
                # Call analyzer with the specific PGN
                analysis_result = analyzer.analyze_game(pgn_text, step_callback=on_move_analyzed)
                result_queue.put({"type": "complete", "analysis": analysis_result})
            except AnalysisCancelled:
                pass
            except Exception as e:
                result_queue.put({"type": "error", "message": str(e)})

        except Exception as e:
            if not stop_event.is_set():
                result_queue.put({"type": "error", "message": str(e)})

    # Start Thread
    bg_thread = threading.Thread(target=analysis_task)
    bg_thread.start()

    def generate():
        current_progress = 0
        try:
            while True:
                item = result_queue.get()
                
                if item.get('type') == 'progress':
                    current_progress += 1
                    yield json.dumps({"type": "progress", "current": current_progress}) + "\n"
                else:
                    yield json.dumps(item) + "\n"
                
                if item.get('type') in ['complete', 'error']:
                    break
        except GeneratorExit:
            stop_event.set()
            raise
        finally:
            stop_event.set()

    return Response(stream_with_context(generate()), mimetype='application/x-ndjson')

if __name__ == '__main__':
    if not os.path.exists('templates'):
        print("❌ Error: 'templates' folder missing.")
    app.run(debug=True, port=5000)
