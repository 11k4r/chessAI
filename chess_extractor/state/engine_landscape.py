import chess
import chess.engine
from dataclasses import dataclass
from typing import Dict, Optional, List

@dataclass
class EngineLandscape:
    """
    Describes the choice architecture of the position, measuring how many 
    good options exist and the risk profile of the current state.
    """
    viable_move_count: int
    pv1_pv2_gap: float
    blunder_risk_index: float
    evaluation_consistency: int
    drawishness_factor: int

    @classmethod
    def extract(cls, board: chess.Board, engine: chess.engine.SimpleEngine, target_depth: int = 24) -> 'EngineLandscape':
        """
        Extracts the landscape features by running a multi-PV engine analysis.
        
        :param board: The current chess.Board.
        :param engine: A running instance of chess.engine.SimpleEngine.
        :param target_depth: The depth limit for the engine search (e.g., 24).
        """
        num_legal_moves = board.legal_moves.count()
        
        # Handle terminal states immediately
        if num_legal_moves == 0:
            return cls(0, 0.0, 0.0, 0, 0)
            
        # CHANGE HERE: Limit the engine by exact depth instead of time
        limit = chess.engine.Limit(depth=target_depth)
        
        # Start async analysis stream to track consistency (how often the top move changes)
        analysis = engine.analysis(board, limit, multipv=num_legal_moves)
        
        best_move_changes = 0
        current_best_move = None
        
        # Dictionary to store the final evaluated score of each move
        final_evals: Dict[chess.Move, float] = {}
        
        for info in analysis:
            if "score" in info and "pv" in info:
                move = info["pv"][0]
                multipv_index = info.get("multipv", 1)
                
                # Convert centipawns to standard pawn floats. Mate is capped at +/- 100.0 pawns.
                score = info["score"].pov(board.turn).score(mate_score=10000) / 100.0
                final_evals[move] = score
                
                # Track Evaluation Consistency (only look at the PV1 line updates)
                if multipv_index == 1:
                    if current_best_move is None:
                        current_best_move = move
                    elif current_best_move != move:
                        best_move_changes += 1
                        current_best_move = move

        # If for some reason the engine didn't return evals, fallback to 0
        if not final_evals:
            return cls(0, 0.0, 0.0, 0, 0)

        # Sort the evaluated moves from best to worst
        sorted_scores = sorted(final_evals.values(), reverse=True)
        top_score = sorted_scores[0]
        
        # 1. Viable Move Count: Moves within 0.50 of the top engine move
        viable_move_count = sum(1 for s in sorted_scores if top_score - s <= 0.50)
        
        # 2. PV1 vs PV2 Gap
        pv1_pv2_gap = (top_score - sorted_scores[1]) if len(sorted_scores) > 1 else 0.0
        
        # 3. Blunder Risk Index: Percentage of moves that drop eval by > 2.00
        blunders = sum(1 for s in sorted_scores if top_score - s > 2.00)
        blunder_risk_index = round(blunders / num_legal_moves, 3)
        
        # 4. Drawishness Factor: Count of moves keeping the ABSOLUTE evaluation between -0.2 and 0.2
        drawish_count = 0
        for score in sorted_scores:
            abs_score = score if board.turn == chess.WHITE else -score
            if -0.20 <= abs_score <= 0.20:
                drawish_count += 1
                
        return cls(
            viable_move_count=viable_move_count,
            pv1_pv2_gap=round(pv1_pv2_gap, 2),
            blunder_risk_index=blunder_risk_index,
            evaluation_consistency=best_move_changes,
            drawishness_factor=drawish_count
        )