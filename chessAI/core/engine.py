import chess.engine
import shutil
import sys
import os

class EngineHandler:
    """
    Manages the external UCI Engine (Stockfish).
    Wrapper to handle asyncio/subprocess issues on Windows+Jupyter.
    """
    def __init__(self, engine_path: str = None, time_limit: float = 0.1):
        # 1. Try to find engine in PATH if not provided
        self.engine_path = engine_path or shutil.which("stockfish")
        self.time_limit = time_limit
        self.engine = None
        
        # Validation: If path provided but not found, warn immediately
        if engine_path and not os.path.exists(engine_path):
            print(f"⚠️ Error: Engine binary not found at {engine_path}")

    def __enter__(self):
        if self.engine_path:
            # CRITICAL UPDATE: We removed the generic try/except block.
            # If the engine fails to start (wrong binary, no permissions), 
            # this will now RAISE an exception so you can see it in the UI/Logs
            # instead of silently failing and returning 0 eval.
            
            # Windows/Jupyter specific fallback check
            try:
                self.engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
            except NotImplementedError:
                # This specific error only happens on Windows with specific event loops
                if sys.platform == 'win32':
                    print("⚠️ Warning: Asyncio subprocesses not supported. Engine disabled.")
                    self.engine = None
                else:
                    raise # Re-raise on Linux/Render
        else:
             print("ℹ️ Note: No engine path provided/found. Dynamic metrics will be skipped.")
             
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.engine:
            try:
                # Using close() is safer/faster for cleanup
                self.engine.close() 
            except Exception as e:
                print(f"DEBUG: Error closing engine: {e}")
                pass

    def analyze(self, board: chess.Board) -> dict:
        """
        Runs a quick analysis of the current board.
        Returns a dictionary with 'score' (cp), 'depth', 'nodes'.
        """
        if not self.engine:
            return {}

        try:
            limit = chess.engine.Limit(time=self.time_limit)
            info = self.engine.analyse(board, limit)
            
            # Normalize score to centipawns (from White's perspective)
            score = info["score"].white()
            score_val = None
            
            if score.is_mate():
                score_val = {"type": "mate", "value": score.mate()}
            else:
                score_val = {"type": "cp", "value": score.score()}

            return {
                "score": score_val,
                "depth": info.get("depth", 0),
                "nodes": info.get("nodes", 0)
            }
        except Exception as e:
            # If analysis fails mid-stream, we log it
            print(f"Analysis Failed: {e}")
            return {}
