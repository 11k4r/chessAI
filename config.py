import os

class Config:
    # Basic App Settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-for-chessai360'
    SITE_NAME = "chessAI 360"
    
    # Server-Side Engine Configuration (Static Analysis)
    # Update this path to where your Stockfish 11 binary lives on the server
    STATIC_ENGINE_PATH = "engines/Stockfish-sf_11/Stockfish-sf_11/src/stockfish.exe"
    
    # Analysis Settings
    Thinking_Time = 1.0  # Seconds for static server-side analysis
    Threads = 2          # CPU threads for server-side engine
    TIME_CONTROL_STYLES = {
        'bullet': 'text-purple-400',    # Purple (Chaotic/Fast)
        'blitz': 'text-yellow-400',     # Yellow (Electric/Lightning)
        'rapid': 'text-blue-400',       # Blue (Calm/Thinking)
        'daily': 'text-cyan-400',       # Cyan (Slow)
        'classical': 'text-cyan-400'
    }