"""
1.x Static Metrics (Bitboard Based)
These metrics rely on O(1) bitwise operations on the board state.
They do not require the adjacency matrix or the engine.
"""
import chess

# --- Constants for Material Calculation ---
VAL_PAWN = 1.0
VAL_BISHOP = 3.25
VAL_QUEEN = 9.75
VAL_BISHOP_PAIR = 0.5

# Dynamic Adjustment Config
BASE_KNIGHT = 3.25
BASE_ROOK = 5.0
ADJ_KNIGHT = 0.0625   # 1/16: Knights like closed positions (more pawns)
ADJ_ROOK = -0.125     # -1/8: Rooks like open positions (fewer pawns)

# --- Constants for Mobility Calculation ---
# Structure: { Phase: { PieceType: Weight, "forward": Bonus, "vertical": Bonus } }
MOBILITY_CONFIG = {
    "Opening": {
        # Knights/Bishops paramount for development. Queens/Rooks passive.
        chess.KNIGHT: 2.2, chess.BISHOP: 2.2, chess.ROOK: 0.5, chess.QUEEN: 0.2,
        "forward_bonus": 0.8,  # High reward for developing forward
        "vertical_bonus": 0.0
    },
    "Middlegame": {
        # Balanced activity. Rooks start to matter.
        chess.KNIGHT: 3.0, chess.BISHOP: 3.0, chess.ROOK: 2.5, chess.QUEEN: 1.5,
        "forward_bonus": 0.3,
        "vertical_bonus": 0.2 # Rooks on open files
    },
    "Endgame": {
        # Rooks and Queens dominate.
        chess.KNIGHT: 3.0, chess.BISHOP: 3.0, chess.ROOK: 4.5, chess.QUEEN: 4.0,
        "forward_bonus": 0.1,  # Forward less strict, mobility in all directions good
        "vertical_bonus": 0.1
    }
}

# --- Masks ---
CENTER_MASK = chess.BB_E4 | chess.BB_D4 | chess.BB_E5 | chess.BB_D5
EXTENDED_CENTER = CENTER_MASK | chess.BB_C3 | chess.BB_F3 | chess.BB_C6 | chess.BB_F6

def calculate_material(ctx) -> dict:
    """
    1.1 Material
    Calculates the material value for both sides using dynamic piece values.
    """
    board = ctx.board
    scores = {}

    for color in [chess.WHITE, chess.BLACK]:
        # 1. Fetch Piece Counts (Fast Bitwise PopCount)
        pawns = len(board.pieces(chess.PAWN, color))
        knights = len(board.pieces(chess.KNIGHT, color))
        bishops = len(board.pieces(chess.BISHOP, color))
        rooks = len(board.pieces(chess.ROOK, color))
        queens = len(board.pieces(chess.QUEEN, color))

        # 2. Calculate Dynamic Values based on Pawn Count
        pawn_diff = pawns - 5
        knight_val = BASE_KNIGHT + (pawn_diff * ADJ_KNIGHT)
        rook_val = BASE_ROOK + (pawn_diff * ADJ_ROOK)

        # 3. Sum Material
        material = (pawns * VAL_PAWN) + \
                   (knights * knight_val) + \
                   (bishops * VAL_BISHOP) + \
                   (rooks * rook_val) + \
                   (queens * VAL_QUEEN)

        # 4. Add Bishop Pair Bonus
        if bishops >= 2:
            material += VAL_BISHOP_PAIR
            
        key = "white" if color == chess.WHITE else "black"
        scores[key] = round(material, 4)

    return scores

def calculate_phase(ctx) -> str:
    """
    1.22 Phase
    Determines game phase: Opening, Middlegame, Endgame.
    Logic:
    - Opening: Move count < 10.
    - Endgame: Both sides have < 14 material (Standard Valuation).
    - Middlegame: All other states.
    """
    board = ctx.board
    
    # 1. Opening Check
    if board.fullmove_number < 10:
        return "Opening"
    
    # 2. Endgame Check (Standard Material < 14 for BOTH sides)
    # Standard Values: P=1, N=3, B=3, R=5, Q=9
    def get_standard_material(color):
        return (len(board.pieces(chess.PAWN, color)) * 1) + \
               (len(board.pieces(chess.KNIGHT, color)) * 3) + \
               (len(board.pieces(chess.BISHOP, color)) * 3) + \
               (len(board.pieces(chess.ROOK, color)) * 5) + \
               (len(board.pieces(chess.QUEEN, color)) * 9)

    white_mat = get_standard_material(chess.WHITE)
    black_mat = get_standard_material(chess.BLACK)
    
    if white_mat < 13 and black_mat < 13:
        return "Endgame"
        
    return "Middlegame"

def calculate_mobility(ctx, phase=None) -> dict:
    """
    1.2 Mobility
    Calculates weighted piece mobility based on Game Phase and Direction.
    """
    board = ctx.board
    if not phase:
        phase = calculate_phase(ctx)
    
    # Default to Middlegame if phase string is unexpected
    weights = MOBILITY_CONFIG.get(phase, MOBILITY_CONFIG["Middlegame"])
    scores = {}

    for color in [chess.WHITE, chess.BLACK]:
        mobility_score = 0.0
        
        # Iterate relevant piece types
        for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            base_weight = weights.get(piece_type, 1.0)
            fwd_bonus = weights.get("forward_bonus", 0.0)
            vert_bonus = weights.get("vertical_bonus", 0.0)
            
            # Get all squares with this piece type
            pieces_sq_set = board.pieces(piece_type, color)
            
            for sq in pieces_sq_set:
                rank = chess.square_rank(sq)
                # Cast to int to ensure bitwise operations work cleanly
                attacks_bb = int(board.attacks(sq))
                
                # --- Directional Logic ---
                if color == chess.WHITE:
                    forward_mask = ~((1 << ((rank + 1) * 8)) - 1) & 0xFFFFFFFFFFFFFFFF
                else:
                    forward_mask = (1 << (rank * 8)) - 1
                
                fwd_moves_bb = attacks_bb & forward_mask
                bwd_moves_bb = attacks_bb & ~forward_mask
                
                fwd_count = bin(fwd_moves_bb).count('1')
                bwd_count = bin(bwd_moves_bb).count('1')
                
                score = (fwd_count * (base_weight + fwd_bonus)) + \
                        (bwd_count * base_weight)
                
                # --- Rook Vertical Bias ---
                if piece_type == chess.ROOK and vert_bonus != 0:
                    file_mask = chess.BB_FILES[chess.square_file(sq)]
                    vert_moves_bb = attacks_bb & file_mask
                    vert_count = bin(vert_moves_bb).count('1')
                    score += (vert_count * vert_bonus)

                mobility_score += score

        key = "white" if color == chess.WHITE else "black"
        scores[key] = round(mobility_score, 2)
        
    return scores

def calculate_pawn_structure(ctx) -> dict:
    """
    Calculates Doubled, Isolated, and Passed pawns.
    Returns counts and a heuristic score.
    """
    board = ctx.board
    scores = {}
    
    for color in [chess.WHITE, chess.BLACK]:
        pawns = board.pieces(chess.PAWN, color)
        opp_pawns = board.pieces(chess.PAWN, not color)
        
        doubled = 0
        isolated = 0
        passed = 0
        connected = 0
        
        # We iterate through files (0-7)
        for file_idx in range(8):
            # 1. Doubled Pawns
            file_mask = chess.BB_FILES[file_idx]
            pawns_on_file = pawns & file_mask
            count = bin(pawns_on_file).count('1')
            if count > 1:
                doubled += (count - 1) 

            # 2. Isolated & Connected Logic
            adj_files = 0
            if file_idx > 0: adj_files |= chess.BB_FILES[file_idx - 1]
            if file_idx < 7: adj_files |= chess.BB_FILES[file_idx + 1]
            
            has_neighbor = (pawns & adj_files)
            if count > 0:
                if not has_neighbor:
                    isolated += count
                else:
                    # Very simplified connected logic: if adjacent file has pawn, count as connected group
                    # Real logic would check ranks, but this is a decent heuristic for static score
                    connected += count

        # 3. Passed Pawns
        for sq in pawns:
            file_idx = chess.square_file(sq)
            rank_idx = chess.square_rank(sq)
            
            if color == chess.WHITE:
                forward_ranks = ~((1 << (8 * (rank_idx + 1))) - 1)
            else:
                forward_ranks = (1 << (8 * rank_idx)) - 1 
            
            span_mask = chess.BB_FILES[file_idx]
            if file_idx > 0: span_mask |= chess.BB_FILES[file_idx - 1]
            if file_idx < 7: span_mask |= chess.BB_FILES[file_idx + 1]
            
            blocker_zone = forward_ranks & span_mask
            if not (opp_pawns & blocker_zone):
                passed += 1

        # Heuristic Score (Higher is better)
        # Passed +20, Connected +5, Isolated -10, Doubled -10
        structure_score = (passed * 20) + (connected * 5) - (isolated * 10) - (doubled * 10)
        # Normalize to 0-100 range roughly, starting from 50 baseline
        final_score = max(0, min(100, 50 + structure_score))

        scores["white" if color == chess.WHITE else "black"] = {
            "score": int(final_score),
            "doubled": doubled, 
            "isolated": isolated, 
            "passed": passed,
            "connected": connected
        }
    return scores

def calculate_space(ctx) -> dict:
    """
    1.8 Space
    Calculates space based on:
    1. Safe Maneuverability: Squares behind own pawns in center files (C,D,E,F) 
       that are NOT attacked by enemy pawns.
       * Added support for B and G files with reduced weight.
    2. Push Bonus: Extra value for pawns advanced to 5th/6th rank.
    """
    board = ctx.board
    scores = {}

    for color in [chess.WHITE, chess.BLACK]:
        key = "white" if color == chess.WHITE else "black"
        
        # --- 1. Identify "Safe Maneuverability" Squares ---
        space_squares = []
        
        enemy_color = not color
        # Get all squares attacked by enemy pawns
        enemy_pawns = board.pieces(chess.PAWN, enemy_color)
        enemy_pawn_attacks = 0
        for sq in enemy_pawns:
            enemy_pawn_attacks |= int(board.attacks(sq)) # Capture set of attacked squares

        # Files and their weights:
        # C(2), D(3), E(4), F(5) -> Weight 1.0
        # B(1), G(6)             -> Weight 0.5 (Reduced Score)
        file_config = {
            2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0,
            1: 0.5, 6: 0.5
        }
        
        total_weighted_score = 0.0
        
        for file_idx, weight in file_config.items():
            # Find front-most pawn for 'color' on this file
            # Force conversion to int to avoid MSB/LSB errors on empty sets
            pawns_on_file = int(board.pieces(chess.PAWN, color) & chess.BB_FILES[file_idx])
            
            if not pawns_on_file:
                # If no pawn on file, we default to standard territory
                front_rank = 4 if color == chess.WHITE else 3 # 0-indexed: Rank 5 or Rank 4 boundary
            else:
                # Get rank of most advanced pawn
                # White: max rank (closest to 7). Black: min rank (closest to 0).
                if color == chess.WHITE:
                    front_rank = chess.square_rank(chess.msb(pawns_on_file))
                else:
                    front_rank = chess.square_rank(chess.lsb(pawns_on_file))
            
            # Iterate ranks "behind" this front rank
            # Using range for ranks behind
            r_range = range(0, front_rank) if color == chess.WHITE else range(7, front_rank, -1)
            
            for r in r_range:
                sq = chess.square(file_idx, r)
                
                # Check 1: On our side? (White: 0-3, Black: 4-7)
                is_own_side = (r < 4) if color == chess.WHITE else (r > 3)
                if not is_own_side:
                    continue
                    
                # Check 2: Not attacked by enemy pawn
                if (1 << sq) & enemy_pawn_attacks:
                    continue
                    
                space_squares.append(sq)
                total_weighted_score += weight
                
        # --- 2. Push Bonus ---
        # Extra value for pawns advanced to 5th or 6th Rank.
        push_bonus = 0
        all_pawns = board.pieces(chess.PAWN, color)
        
        if color == chess.WHITE:
            bonus_ranks = chess.BB_RANK_5 | chess.BB_RANK_6
        else:
            bonus_ranks = chess.BB_RANK_4 | chess.BB_RANK_3
            
        advanced_pawns = int(all_pawns) & bonus_ranks
        push_bonus = bin(advanced_pawns).count('1')
        
        # Total Score (Rounded to int for display consistency, or keep float?)
        # Since other metrics are ints/floats, let's keep it as a float or round to 1 decimal.
        # push_bonus is 1.0 per pawn implicitly.
        
        final_score = round(total_weighted_score + push_bonus, 1)
        
        scores[key] = {
            "score": final_score,
            "space_squares": space_squares # For UI highlighting
        }

    return scores


def calculate_activity(ctx) -> dict:
    """
    1.10 Activity
    Heuristic: (Mobility * 0.5) + (Center Control * 2.0)
    """
    mobility = calculate_mobility(ctx)
    center = calculate_center_control(ctx)
    
    return {
        "white": round(mobility["white"] * 0.5 + center["white"] * 2.0, 2),
        "black": round(mobility["black"] * 0.5 + center["black"] * 2.0, 2)
    }

import chess

def calculate_center_control(ctx) -> dict:
    board = ctx.board
    scores = { "white": 0.0, "black": 0.0 }

    # 1. Define Zones
    primary_center = [chess.E4, chess.D4, chess.E5, chess.D5]
    
    # Extended center: c3-f6 rectangle excluding primary
    extended_center = []
    
    # --- FIX: Use Integers directly ---
    # Files C, D, E, F correspond to indices 2, 3, 4, 5
    files = [2, 3, 4, 5]
    
    # Ranks 3, 4, 5, 6 correspond to indices 2, 3, 4, 5
    ranks = [2, 3, 4, 5]
    # ----------------------------------

    for f in files:
        for r in ranks:
            sq = chess.square(f, r)
            if sq not in primary_center:
                extended_center.append(sq)

    # Pre-fetch bitboards
    pawns_w = int(board.pieces(chess.PAWN, chess.WHITE))
    pawns_b = int(board.pieces(chess.PAWN, chess.BLACK))

    # 2. Analysis Helper
    def analyze_square(sq, zone_weight):
        for color in [chess.WHITE, chess.BLACK]:
            key = "white" if color == chess.WHITE else "black"
            my_pawns = pawns_w if color == chess.WHITE else pawns_b
            
            # --- A. CONTROL SCORE ---
            attackers = int(board.attackers(color, sq))
            pawn_attackers = attackers & my_pawns
            piece_attackers = attackers & ~my_pawns
            
            control_score = (bin(pawn_attackers).count('1') * 1.0) + \
                            (bin(piece_attackers).count('1') * 0.75)

            # --- B. OCCUPATION SCORE ---
            occupation_score = 0.0
            piece_on_sq = board.piece_at(sq)
            
            if piece_on_sq and piece_on_sq.color == color:
                if piece_on_sq.piece_type == chess.PAWN:
                    val = 2.0
                elif piece_on_sq.piece_type in [chess.KNIGHT, chess.BISHOP]:
                    val = 1.5
                else:
                    val = 0.2 
                
                # Stability Check
                if pawn_attackers: 
                    if piece_on_sq.piece_type == chess.PAWN:
                        val *= 2.0 # Pawn Chain
                    else:
                        val *= 1.5 # Supported Outpost
                
                occupation_score = val

            # --- C. TOTAL SQUARE SCORE ---
            square_total = (occupation_score + control_score) * zone_weight
            scores[key] += square_total

    # 3. Iterate Zones
    for sq in primary_center:
        analyze_square(sq, zone_weight=2.0)

    for sq in extended_center:
        analyze_square(sq, zone_weight=1.0)

    scores["white"] = round(scores["white"], 2)
    scores["black"] = round(scores["black"], 2)

    return scores


def calculate_key_square_control(ctx) -> dict:
    """
    1.19 Key Square Control
    Focuses on OUTPOSTS for Knights.
    Criteria: Knight on Rank 4,5,6 (relative), protected by pawn, no enemy pawn can attack.
    """
    board = ctx.board
    scores = {"white": 0, "black": 0}
    
    for color in [chess.WHITE, chess.BLACK]:
        key = "white" if color == chess.WHITE else "black"
        knights = board.pieces(chess.KNIGHT, color)
        pawns = board.pieces(chess.PAWN, color)
        enemy_pawns = board.pieces(chess.PAWN, not color)
        
        outpost_score = 0
        
        if color == chess.WHITE:
            outpost_ranks = [3, 4, 5] 
        else:
            outpost_ranks = [4, 3, 2]
            
        for sq in knights:
            rank = chess.square_rank(sq)
            if rank not in outpost_ranks:
                continue
                
            # Check 1: Protected by Pawn
            defenders = board.attackers(color, sq)
            if not (defenders & pawns):
                continue
                
            # Check 2: No enemy pawn controls it
            attackers = board.attackers(not color, sq)
            if attackers & enemy_pawns:
                continue
            
            outpost_score += 1
            
        scores[key] = outpost_score

    return scores

def calculate_color_complex_control(ctx) -> dict:
    """
    1.20 Color Complex Control
    Analyzes control of Light vs Dark squares.
    """
    board = ctx.board
    scores = {"white": {}, "black": {}}
    
    for color in [chess.WHITE, chess.BLACK]:
        key = "white" if color == chess.WHITE else "black"
        
        combined_attacks = 0
        # FIX: Iterate over occupied squares of that color
        occupied_bb = board.occupied_co[color]
        for sq in chess.SquareSet(occupied_bb):
             combined_attacks |= int(board.attacks(sq))
             
        light = bin(combined_attacks & chess.BB_LIGHT_SQUARES).count('1')
        dark = bin(combined_attacks & chess.BB_DARK_SQUARES).count('1')
        
        scores[key] = {"light": light, "dark": dark}
        
    return scores

def calculate_king_activity(ctx) -> dict:
    """
    1.21 King Activity
    Endgame metric. Measures King's proximity to center.
    Returns 0.0 if not Endgame.
    """
    phase = calculate_phase(ctx)
    if phase != "Endgame":
        return {"white": 0.0, "black": 0.0}
        
    board = ctx.board
    scores = {}
    
    for color in [chess.WHITE, chess.BLACK]:
        key = "white" if color == chess.WHITE else "black"
        king_sq = board.king(color)
        if king_sq is None: 
            scores[key] = 0.0
            continue
            
        rank, file = chess.square_rank(king_sq), chess.square_file(king_sq)
        
        # Normalize 0..7 to distance from center (0 is edge, 3 is center)
        rank_centrality = 3.5 - abs(rank - 3.5)
        file_centrality = 3.5 - abs(file - 3.5)
        
        scores[key] = round(rank_centrality + file_centrality, 2)
        
    return scores
    
    
def calculate_novelty(ctx) -> int:
    """
    Returns a 'Commonality Score' (0-100).
    100 = Very Common (Opening Book)
    0 = Unique / Novelty
    
    Without a real database, we use a heuristic based on move number.
    Games usually leave book around move 10-15.
    """
    move_num = ctx.board.fullmove_number
    
    if move_num <= 5: return 100
    if move_num <= 10: return 80
    if move_num <= 15: return 40
    if move_num <= 20: return 10
    return 0