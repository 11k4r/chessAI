"""
1.x Graph Metrics (Matrix Based)
These metrics rely on the 64x64 Interaction Matrices (Attack/Defense).
They calculate relationships between pieces.
"""
import numpy as np
import chess

def calculate_connectivity(ctx) -> dict:
    """
    1.3 Connectivity
    Measures how well pieces defend each other.
    Calculated as the number of defense links between friendly pieces.
    """
    board = ctx.board
    scores = {}

    for color in [chess.WHITE, chess.BLACK]:
        key = "white" if color == chess.WHITE else "black"
        
        # Get the attack matrix for this color (Row=Attacker, Col=Target)
        matrix = ctx.white_attacks if color == chess.WHITE else ctx.black_attacks
        
        # We only care about columns (targets) that contain FRIENDLY pieces
        # Create a mask of squares occupied by 'color'
        # FIX: Cast integer bitboard to SquareSet to iterate squares
        friendly_squares = list(chess.SquareSet(board.occupied_co[color]))
        
        if not friendly_squares:
            scores[key] = 0
            continue
            
        # Slice matrix: All rows, only Friendly Columns
        defense_submatrix = matrix[:, friendly_squares]
        
        # Sum of links
        scores[key] = int(np.sum(defense_submatrix))

    return scores

def calculate_attack(ctx) -> dict:
    """
    1.4 Attack
    Measures the intensity of attacks on enemy squares and empty squares.
    """
    scores = {}
    for color in [chess.WHITE, chess.BLACK]:
        key = "white" if color == chess.WHITE else "black"
        matrix = ctx.white_attacks if color == chess.WHITE else ctx.black_attacks
        scores[key] = int(np.sum(matrix)) # Total number of attacks generated
    return scores

def calculate_defence(ctx) -> dict:
    """
    1.5 Defence
    Average number of defenders per piece.
    """
    board = ctx.board
    scores = {}
    
    conn = calculate_connectivity(ctx)
    
    for color in [chess.WHITE, chess.BLACK]:
        key = "white" if color == chess.WHITE else "black"
        # FIX: Use bin().count('1') for bit counting on integer bitboards to be safe across python versions
        piece_count = bin(board.occupied_co[color]).count('1')
        if piece_count > 0:
            scores[key] = round(conn[key] / piece_count, 2)
        else:
            scores[key] = 0.0
            
    return scores

def calculate_king_safety(ctx) -> dict:
    """
    1.6 King Safety
    Analyzes checks and attacks around the king.
    Lower score is safer (0 is perfect safety).
    """
    board = ctx.board
    scores = {}

    for color in [chess.WHITE, chess.BLACK]:
        key = "white" if color == chess.WHITE else "black"
        king_sq = board.king(color)
        if king_sq is None:
            scores[key] = 100.0 # Maximum danger if no king (shouldn't happen)
            continue

        # 1. Check Status
        in_check = 10.0 if board.is_check() and board.turn == color else 0.0
        
        # 2. Attacks around King
        # Get enemy attack matrix
        enemy_matrix = ctx.black_attacks if color == chess.WHITE else ctx.white_attacks
        
        # Define King Zone (King + adjacent squares)
        # board.attacks(sq) returns bitboard int.
        king_zone_bb = board.attacks(king_sq) | (1 << king_sq)
        
        # FIX: Efficiently convert bitboard to list of indices
        king_zone_sqs = list(chess.SquareSet(king_zone_bb))
        
        # Sum attacks targeting the king zone
        zone_attacks = np.sum(enemy_matrix[:, king_zone_sqs])
        
        scores[key] = round(in_check + float(zone_attacks), 2)

    return scores

def calculate_weakness(ctx) -> dict:
    """
    1.11 Weakness
    Identifies number of undefended pieces (Hanging pieces).
    """
    board = ctx.board
    scores = {}

    for color in [chess.WHITE, chess.BLACK]:
        key = "white" if color == chess.WHITE else "black"
        
        # Matrix of own attacks
        matrix = ctx.white_attacks if color == chess.WHITE else ctx.black_attacks
        
        hanging_count = 0
        # Iterate friendly pieces
        # FIX: ensure SquareSet is used
        for sq in chess.SquareSet(board.occupied_co[color]):
            # Check column sum (how many incoming friendly edges?)
            defenders = np.sum(matrix[:, sq])
            if defenders == 0:
                hanging_count += 1
                
        scores[key] = hanging_count

    return scores

def calculate_harmony(ctx) -> dict:
    """
    1.12 Harmony
    Heuristic: Connectivity / Number of Pieces.
    High harmony means the army is well-knit.
    """
    return calculate_defence(ctx) # Logic is identical to Defence avg

def calculate_piece_quality(ctx) -> dict:
    """
    1.15 Piece Quality
    Ratio of 'Good Squares' vs 'Bad Squares'.
    """
    # Placeholder for complex logic. 
    # For now, return normalized activity.
    return {k: round(v/50.0, 2) for k,v in calculate_attack(ctx).items()}