import chess
import chess.pgn
import io
import json
import math
import re

from .core.context import AnalysisContext
from .core.engine import EngineHandler
from .metrics import static, graph, dynamic
from .aggregators.game_stats import GameStatsCalculator

class ChessAnalyzer:
    """
    The Main Interface for the chessAI package.
    Orchestrates the analysis pipeline.
    """
    
    def __init__(self, engine_path: str = None):
        self.engine_path = engine_path

    def analyze_position(self, fen: str, engine_handler: EngineHandler = None) -> dict:
        """
        Analyzes a single position (FEN).
        Returns a JSON-serializable dictionary of all 1.x metrics.
        """
        board = chess.Board(fen)
        
        # 1. Run Engine (Dynamic Analysis)
        engine_info = {}
        if engine_handler:
            # For position analysis, we just want the score of the current pos
            engine_info = engine_handler.analyze(board)

        # 2. Build Context (Graph & Bitboards)
        ctx = AnalysisContext(board, engine_info)
        
        # 3. Calculate Metrics
        metrics = {
            "fen": fen,
            "static": {
                "material": static.calculate_material(ctx),
                "mobility": static.calculate_mobility(ctx),
                "pawn_structure": static.calculate_pawn_structure(ctx),
                "space": static.calculate_space(ctx),
                "phase": static.calculate_phase(ctx),
                "activity": static.calculate_activity(ctx),
                "center_control": static.calculate_center_control(ctx),
                "key_squares": static.calculate_key_square_control(ctx),
                "color_complex": static.calculate_color_complex_control(ctx),
                "king_activity": static.calculate_king_activity(ctx),
                "novelty": static.calculate_novelty(ctx)
            },
            "graph": {
                "connectivity": graph.calculate_connectivity(ctx),
                "attack": graph.calculate_attack(ctx),
                "defence": graph.calculate_defence(ctx),
                "king_safety": graph.calculate_king_safety(ctx),
                "harmony": graph.calculate_harmony(ctx),
                "weakness": graph.calculate_weakness(ctx)
            },
            "dynamic": {
                "eval": dynamic.calculate_eval(ctx),
                "complexity": dynamic.calculate_complexity(ctx),
                "criticality": dynamic.calculate_criticality(ctx),
                "threats": dynamic.calculate_threats(ctx)
            }
        }
        
        return metrics

    def analyze_game(self, pgn_string: str, step_callback=None, engine_handler: EngineHandler = None) -> dict:
        """
        Analyzes a full game PGN.
        Calculates Move Classification (Best/Blunder/etc) and Game Stats.
        """
        pgn_io = io.StringIO(pgn_string)
        game = chess.pgn.read_game(pgn_io)
        
        if game is None:
            return {"error": "Invalid PGN string"}

        # Internal helper to run the loop with a guaranteed engine
        def _execute_analysis(engine):
            board = game.board()
            positions_data = []
            
            # --- 1. Analyze Initial Position ---
            start_metrics = self.analyze_position(board.fen(), engine)
            # Add dummy move info for start
            start_metrics['move_info'] = {"san": "Start", "time_spend": "0s"}
            positions_data.append(start_metrics)
            
            # Track previous eval for classification
            prev_eval_val = 0.0 
            
            # Map classifications to numeric scores
            accuracy_scores = {
                "Best": 100, "Excellent": 95, "Good": 75,
                "Inaccuracy": 40, "Mistake": 10, "Blunder": 0
            }
            
            # Parse Time Control for Increment
            increment = 0.0
            time_control = game.headers.get("TimeControl", "")
            initial_time = 0.0
            
            if "+" in time_control:
                try:
                    init_str, inc_str = time_control.split("+")
                    increment = float(inc_str)
                    initial_time = float(init_str)
                except:
                    pass
            elif time_control.isdigit():
                 initial_time = float(time_control)
                 
            # State tracking for clocks
            last_white_clock = initial_time
            last_black_clock = initial_time
            
            # --- 2. Iterate Moves (Using Nodes to get clock) ---
            node = game
            move_count = 0
            while node.variations:
                next_node = node.variation(0)
                move = next_node.move
                san = board.san(move)
                move_count += 1
                
                # --- Time Calculation ---
                time_spend = "-" 
                clk_current = next_node.clock()
                
                # Determine who moved and calculate delta against THEIR last clock
                is_white = (board.turn == chess.WHITE)
                
                if clk_current is not None:
                    prev_clock = last_white_clock if is_white else last_black_clock
                    
                    # Time Spent = Prev - Curr
                    # Note: Increment is usually added *after* the move.
                    # PGN clock is "Time remaining".
                    # If I start with 300. I take 5s. Clock becomes 295.
                    # If increment is 2s. Clock becomes 297.
                    # Delta = 300 - 297 = 3s.
                    # Real time spent = 5s.
                    # Formula: Spent = (Prev_Clock - Curr_Clock) + Increment
                    
                    delta = (prev_clock - clk_current) + increment
                    final_time = max(0.0, delta)
                    
                    # Update the state for next time
                    if is_white:
                        last_white_clock = clk_current
                    else:
                        last_black_clock = clk_current

                    if final_time >= 60:
                        m = int(final_time // 60)
                        s = int(final_time % 60)
                        time_spend = f"{m}m {s}s"
                    else:
                        time_spend = f"{round(final_time, 1)}s"
                
                # Update Board
                board.push(move)
                
                # Analyze New Position
                pos_metrics = self.analyze_position(board.fen(), engine)
                
                # Extract Evaluation
                curr_eval_data = pos_metrics['dynamic']['eval']
                curr_cp = self._normalize_score(curr_eval_data)
                
                # Calculate Move Classification
                mover = not board.turn 
                if mover == chess.WHITE:
                    delta_score = prev_eval_val - curr_cp 
                else:
                    delta_score = curr_cp - prev_eval_val 
                
                classification = self._classify_move(delta_score)
                accuracy_val = accuracy_scores.get(classification, 0)
                
                # Add metadata
                pos_metrics['move_info'] = {
                    "san": san,
                    "turn": "White" if mover == chess.WHITE else "Black",
                    "eval_delta": round(delta_score, 2),
                    "classification": classification,
                    "accuracy": accuracy_val,
                    "time_spend": time_spend
                }
                pos_metrics['move_classification'] = classification
                positions_data.append(pos_metrics)
                prev_eval_val = curr_cp
                
                # Advance Node
                node = next_node
                
                # Progress Hook
                if step_callback:
                    step_callback()
            
            return positions_data

        # Decision: Use provided handler or create new context
        if engine_handler:
            positions_data = _execute_analysis(engine_handler)
        else:
            with EngineHandler(self.engine_path) as engine:
                positions_data = _execute_analysis(engine)

        # 3. Aggregate Game Stats
        game_metrics = GameStatsCalculator.calculate(positions_data, game.headers.get("Result", "*"))

        return {
            "headers": dict(game.headers),
            "game_metrics": game_metrics,
            "positions": positions_data 
        }

    def _normalize_score(self, score_data: dict) -> float:
        """Converts mate scores to large CP values."""
        if score_data['type'] == 'mate':
            val = score_data['value']
            return 1000.0 if val > 0 else -1000.0
        return float(score_data['value'])

    def _classify_move(self, delta_loss: float) -> str:
        """Classifies move based on Centipawn Loss (delta)."""
        if delta_loss <= 10: return "Best"
        if delta_loss <= 25: return "Excellent"
        if delta_loss <= 50: return "Good"
        if delta_loss <= 100: return "Inaccuracy"
        if delta_loss <= 300: return "Mistake"
        return "Blunder"