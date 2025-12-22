"""
1.x Dynamic Metrics (Engine Based)
These metrics require an external UCI engine (Stockfish).
"""
import chess

def calculate_eval(ctx) -> dict:
    """
    1.9 Eval
    Returns the engine evaluation in centipawns or mate score.
    Returns: {"type": "cp"|"mate", "value": int}
    """
    if not ctx.engine_info or "score" not in ctx.engine_info:
        return {"type": "cp", "value": 0}
    return ctx.engine_info["score"]

def calculate_initiative(ctx) -> dict:
    """
    1.13 Initiative
    Measures who is dictating the flow.
    Heuristic: Side to move has initiative if Eval is good + High Attack stat.
    """
    # Placeholder: Usually requires comparing previous move eval to current.
    # We will return 0 for single position analysis.
    return {"white": 0.0, "black": 0.0}

def calculate_complexity(ctx) -> float:
    """
    1.16 Complexity
    Measures the 'chaos' of the position.
    Heuristic based on:
    1. Material Imbalance
    2. Number of legal moves (Branching factor)
    3. Pawn Tension (attacks between pawns)
    """
    board = ctx.board
    
    # 1. Branching Factor
    legal_moves = board.legal_moves.count()
    
    # 2. Pawn Tension (Opposing pawns attacking each other)
    w_pawns = board.pieces(chess.PAWN, chess.WHITE)
    b_pawns = board.pieces(chess.PAWN, chess.BLACK)
    
    tension = 0
    for sq in w_pawns:
        attacks = board.attacks(sq)
        if attacks & b_pawns:
            tension += 1
            
    # Normalize score (0-10 scale approximation)
    complexity = (legal_moves / 10.0) + (tension * 2.0)
    return round(complexity, 2)

def calculate_criticality(ctx) -> float:
    """
    1.17 Criticality
    Determines if the position is forced/sharp.
    """
    # Heuristic: Check + Low number of legal moves = Critical
    board = ctx.board
    legal_moves = board.legal_moves.count()
    is_check = board.is_check()
    
    score = 0.0
    if is_check:
        score += 5.0
    if legal_moves < 5:
        score += (5 - legal_moves) * 2.0
        
    return min(10.0, score)

def calculate_threats(ctx) -> list:
    """
    1.18 Threats
    Identifies immediate tactical threats.
    """
    # In a full implementation, we would do a null-move search.
    # Here we check immediate checks/captures available to the side to move.
    board = ctx.board
    threats = []
    
    for move in board.legal_moves:
        if board.gives_check(move):
            threats.append(board.san(move) + "+")
        elif board.is_capture(move):
            # Simple heuristic: Only list captures of higher value pieces
            # (Skipped for brevity, listing all captures)
            threats.append(board.san(move))
            
    # Return top 5 threats to keep JSON small
    return threats[:5]