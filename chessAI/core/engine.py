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
        
        if not self.engine_path:
            # We don't print here to avoid spamming if user didn't want engine
            pass

    def __enter__(self):
        if self.engine_path:
            try:
                # Attempt to start the engine
                self.engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
            except NotImplementedError:
                # Specific catch for Windows Jupyter environment issue
                print("⚠️ Warning: Engine analysis disabled.")
                print("   Reason: 'asyncio' subprocesses are not supported in this Windows Jupyter environment.")
                print("   Fix: Run as a standalone script or assume static metrics only.")
                self.engine = None
            except Exception as e:
                print(f"⚠️ Warning: Could not start engine at '{self.engine_path}'.")
                print(f"   Error: {e}")
                self.engine = None
        else:
             print("ℹ️ Note: No engine path provided/found. Dynamic metrics (eval, classification) will be skipped.")
             
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.engine:
            try:
                # Using close() instead of quit() is often safer in 
                # problematic environments (like Windows Jupyter) 
                # because it terminates the process/transport immediately 
                # rather than waiting for a polite UCI 'quit' response.
                print("DEBUG: Closing Engine...")
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
            # This is critical for consistent game analysis graphs
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
            # Fallback if engine dies mid-analysis
            return {}