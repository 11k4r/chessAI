import pandas as pd
import math
import chess
from typing import Dict, List, Optional
from core.static_engine import get_engine
from core.parser import parse_trace
import subprocess
import numpy as np
import pandas as pd
from typing import Dict, Set, List, Any

def extract_static_features(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Calculates a comprehensive set of static evaluation features from a Stockfish 11 trace,
    EXCLUDING Material scores.

    Returns a nested dictionary:
    {
        'FeatureName': {
            'White': float,
            'Black': float,
            'Diff': float  # (White - Black)
        },
        ...
    }
    """
    if df.empty:
        return {}

    # --- Constants & Configuration ---
    DIV = 213.0  # Normalization: 213 internal units ~ 1 Pawn

    # Square Definitions for Center Control
    CENTER_SQUARES = {'e4', 'd4', 'e5', 'd5'}
    EXTENDED_CENTER = {
        'c3', 'd3', 'e3', 'f3',
        'c4', 'f4',
        'c5', 'f5',
        'c6', 'd6', 'e6', 'f6'
    }
    
    # Heuristic Weights (internal CP units) for Center Control
    W_CENTER_PAWN = 30.0
    W_EXTENDED_PAWN = 10.0

    # --- 1. Global State Extraction (Phase & Scale Factor) ---
    phase_row = df[df['Feature'] == 'Game Phase']
    phase = float(phase_row['Value'].iloc[0]) if not phase_row.empty else 128.0

    sf_row = df[df['Feature'] == 'Scale Factor']
    sf_raw = float(sf_row['Value'].iloc[0]) if not sf_row.empty else 64.0
    sf = sf_raw / 64.0

    # --- 2. Helper Functions ---
    def get_interpolated_score(mg: float, eg: float) -> float:
        """Stockfish Tapered Eval Formula"""
        return (mg * phase + (eg * (128.0 - phase) * sf)) / 128.0

    def get_val(feat_name: str, color: str) -> float:
        """Fetches and interpolates the score for a single feature/color."""
        rows = df[(df['Feature'] == feat_name) & (df['Color'] == color)]
        if rows.empty: 
            return 0.0
        mg = rows[rows['Phase'] == 'MG']['Value'].sum()
        eg = rows[rows['Phase'] == 'EG']['Value'].sum()
        return get_interpolated_score(mg, eg)

    def get_squares(feat_names: List[str], color: str) -> Set[str]:
        """Collects unique squares from metadata lists for a group of features."""
        sqs = set()
        for feat in feat_names:
            rows = df[(df['Feature'] == feat) & (df['Color'] == color)]
            if 'metadata' in rows.columns:
                for meta in rows['metadata']:
                    if isinstance(meta, list):
                        sqs.update(meta)
        return sqs

    # Initialize the results container
    results = {}

    # Helper to add result in the desired format
    def add_feature_result(feature_name: str, white_val: float, black_val: float):
        results[feature_name] = {
            'White': white_val,
            'Black': black_val,
            'Diff': white_val - black_val
        }

    # --- 3. Standard Stockfish Features Calculation ---
    sf_features_map = {
        'Mobility': [
            'Knight Mobility', 'Bishop Mobility', 'Rook Mobility', 'Queen Mobility'
        ],
        'KingSafety': [
            'Shelter Base', 'Shelter Str', 'Storm Penalty', 'Proximity',
            'Danger', 'Pawnless Flank', 'Flank Attacks'
        ],
        'PawnStructure': [
            'Isolated', 'Backward', 'Doubled', 'Connected', 
            'Weak Lever', 'Weak Unopposed',
            'Passed Rank', 'Passed Prox', 'Passed Block', 'Passed File'
        ],
        'Space': ['Space'],
        'Threats': [
            'Threat Minor', 'Threat Rook', 'Threat King', 'Hanging', 
            'Restricted', 'Safe Pawn', 'Pawn Push', 
            'Knight On Q', 'Slider On Q'
        ]
    }

    for category, atoms in sf_features_map.items():
        scores = {'White': 0.0, 'Black': 0.0}
        
        for color in ['White', 'Black']:
            mg_total = 0.0
            eg_total = 0.0
            for atom in atoms:
                rows = df[(df['Feature'] == atom) & (df['Color'] == color)]
                if not rows.empty:
                    mg_total += rows[rows['Phase'] == 'MG']['Value'].sum()
                    eg_total += rows[rows['Phase'] == 'EG']['Value'].sum()
            
            final_score = get_interpolated_score(mg_total, eg_total)
            scores[color] = final_score / DIV
        
        add_feature_result(category, scores['White'], scores['Black'])

    # --- 4. Extended Strategic Features Calculation ---

    # Feature Group Definitions
    group_activity = [
        'Knight Mobility', 'Bishop Mobility', 'Rook Mobility', 'Queen Mobility',
        'Knight Outpost', 'Bishop Outpost', 'Rook Open File', 'Bishop Long Diag'
    ]
    group_clumsiness = [
        'Rook Trapped', 'Blocked', 'Bishop Cornered', 
        'Knight Protector', 'Bishop Protector', 'Bishop Pawn Penalty'
    ]
    group_complexity = [
        'Imbalance', 'Hanging', 'Winnable Total', 'Danger', 'Bishop King Ring'
    ]
    group_weakness = [
        'Isolated', 'Backward', 'Doubled', 'Weak Unopposed', 'Weak Lever',
        'Weak Queen', 'Trapped Rook', 'Pawnless Flank'
    ]
    group_aggression = [
        'Threat King', 'Storm Penalty', 'Flank Attacks', 'Threat Minor', 
        'Threat Rook', 'Pawn Push', 'Slider On Q', 'Knight On Q'
    ]
    group_integrity_pos = [
        'Shelter Base', 'Shelter Str', 'Connected', 'Space', 
        'Passed Rank', 'Passed Prox', 'Passed Block'
    ]
    group_integrity_neg = [
        'Isolated', 'Backward', 'Doubled', 'Pawnless Flank', 'Hanging'
    ]

    # Pre-fetch Mobility scores (needed for Squeeze)
    mobility_scores = {}
    for c in ['White', 'Black']:
        mobility_scores[c] = sum(get_val(f, c) for f in [
            'Knight Mobility', 'Bishop Mobility', 'Rook Mobility', 'Queen Mobility'
        ])

    # Temporary storage for extended features to group by color
    extended_features = {
        'CenterControl': {'White': 0.0, 'Black': 0.0},
        'Activity': {'White': 0.0, 'Black': 0.0},
        'Harmony': {'White': 0.0, 'Black': 0.0},
        'Complexity': {'White': 0.0, 'Black': 0.0},
        'Weakness': {'White': 0.0, 'Black': 0.0},
        'Aggression': {'White': 0.0, 'Black': 0.0},
        'Integrity': {'White': 0.0, 'Black': 0.0},
        'Squeeze': {'White': 0.0, 'Black': 0.0}
    }

    for color in ['White', 'Black']:
        opp_color = 'Black' if color == 'White' else 'White'
        
        # A. Center Control
        center_score = get_val('Space', color)
        pawn_feats = ['Isolated', 'Backward', 'Doubled', 'Connected', 'Weak Lever', 'Passed Rank']
        pawn_sqs = get_squares(pawn_feats, color)
        center_pawns = len([s for s in pawn_sqs if s in CENTER_SQUARES])
        ext_center_pawns = len([s for s in pawn_sqs if s in EXTENDED_CENTER])
        center_score += (center_pawns * W_CENTER_PAWN) + (ext_center_pawns * W_EXTENDED_PAWN)
        extended_features['CenterControl'][color] = center_score / DIV

        # B. Activity
        activity = sum(get_val(f, color) for f in group_activity)
        extended_features['Activity'][color] = activity / DIV

        # C. Harmony
        clumsiness = sum(abs(get_val(f, color)) for f in group_clumsiness)
        extended_features['Harmony'][color] = (activity - clumsiness) / DIV

        # D. Complexity
        complexity = sum(abs(get_val(f, color)) for f in group_complexity)
        extended_features['Complexity'][color] = complexity / DIV

        # E. Weakness
        weakness = sum(abs(get_val(f, color)) for f in group_weakness)
        extended_features['Weakness'][color] = weakness / DIV

        # F. Aggression
        aggression = sum(get_val(f, color) for f in group_aggression)
        winnable = get_val('Winnable Total', color)
        if winnable > 0: 
            aggression += winnable
        extended_features['Aggression'][color] = aggression / DIV

        # G. Integrity
        pos_integrity = sum(get_val(f, color) for f in group_integrity_pos)
        neg_integrity = sum(abs(get_val(f, color)) for f in group_integrity_neg)
        extended_features['Integrity'][color] = (pos_integrity - neg_integrity) / DIV

        # H. Squeeze
        space = get_val('Space', color)
        restricted = get_val('Restricted', color)
        mob_diff = mobility_scores[color] - mobility_scores[opp_color]
        extended_features['Squeeze'][color] = (space + restricted + mob_diff) / DIV

    # Add extended features to final results
    for feat_name, scores in extended_features.items():
        add_feature_result(feat_name, scores['White'], scores['Black'])

    return results




import chess

def calculate_material_and_imbalance(fen):

    board = chess.Board(fen)
    # --- 1. Constants (from types.h) ---
    DIV = 213.0
    
    # Piece Values [MG, EG]
    PIECE_VALS = {
        chess.PAWN:   (128,  213),
        chess.KNIGHT: (781,  854),
        chess.BISHOP: (825,  915),
        chess.ROOK:   (1276, 1380),
        chess.QUEEN:  (2538, 2682),
        chess.KING:   (0, 0)
    }

    # Phase Limits
    MIDGAME_LIMIT = 15258
    ENDGAME_LIMIT = 3915
    PHASE_MIDGAME = 128

    # Imbalance Tables (from material.cpp)
    # Indices: 0=BishopPair, 1=Pawn, 2=Knight, 3=Bishop, 4=Rook, 5=Queen
    Q_OURS = [
        [1438],                                      
        [  40,   38],                                
        [  32,  255, -62],                           
        [   0,  104,   4,    0],                     
        [ -26,   -2,  47,   105,  -208],             
        [-189,   24, 117,   133,  -134, -6]          
    ]
    
    Q_THEIRS = [
        [   0],                                      
        [  36,    0],                                
        [   9,   63,   0],                           
        [  59,   65,  42,     0],                    
        [  46,   39,  24,   -24,    0],              
        [  97,  100, -42,   137,  268,    0]         
    ]

    # --- 2. Helper Functions ---
    def get_counts(color):
        return [
            1 if len(board.pieces(chess.BISHOP, color)) > 1 else 0, # Bishop Pair
            len(board.pieces(chess.PAWN, color)),
            len(board.pieces(chess.KNIGHT, color)),
            len(board.pieces(chess.BISHOP, color)),
            len(board.pieces(chess.ROOK, color)),
            len(board.pieces(chess.QUEEN, color))
        ]

    def calc_raw(color):
        mg, eg = 0, 0
        npm = 0 # Non-pawn material for phase calc
        
        for pt, (v_mg, v_eg) in PIECE_VALS.items():
            count = len(board.pieces(pt, color))
            mg += count * v_mg
            eg += count * v_eg
            if pt != chess.PAWN and pt != chess.KING:
                npm += count * v_mg # SF uses MG values for Phase calc
        return mg, eg, npm

    def calc_imbalance_side(us_counts, them_counts):
        bonus = 0
        for pt1 in range(6):
            if us_counts[pt1] == 0: continue
            v = 0
            for pt2 in range(pt1 + 1):
                v += Q_OURS[pt1][pt2] * us_counts[pt2] + Q_THEIRS[pt1][pt2] * them_counts[pt2]
            bonus += us_counts[pt1] * v
        return bonus

    # --- 3. Execute Calculations ---
    
    # Raw Material & Phase
    w_mg, w_eg, w_npm = calc_raw(chess.WHITE)
    b_mg, b_eg, b_npm = calc_raw(chess.BLACK)
    
    # Calculate Phase
    # npm = non-pawn material of both sides clamped
    npm_total = max(ENDGAME_LIMIT, min(MIDGAME_LIMIT, w_npm + b_npm))
    phase = int(((npm_total - ENDGAME_LIMIT) * PHASE_MIDGAME) / (MIDGAME_LIMIT - ENDGAME_LIMIT))
    
    # Imbalance
    w_counts = get_counts(chess.WHITE)
    b_counts = get_counts(chess.BLACK)
    
    w_imb_val = calc_imbalance_side(w_counts, b_counts)
    b_imb_val = calc_imbalance_side(b_counts, w_counts)
    
    # In SF, imbalance is (White - Black) / 16.
    # We assign the split values to each side.
    # Note: Imbalance applies to both MG and EG in Stockfish evaluation.
    w_imb_score = w_imb_val / 16.0
    b_imb_score = b_imb_val / 16.0
    
    # --- 4. Totals (Interpolated) ---
    
    def interpolate(mg, eg):
        # Using ScaleFactor=64 (1.0) as default since we don't have full SF endgame tables
        sf = 1.0 
        return (mg * phase + (eg * (128.0 - phase) * sf)) / 128.0

    # White Total Material
    w_total_mg = w_mg + w_imb_score
    w_total_eg = w_eg + w_imb_score
    w_final = interpolate(w_total_mg, w_total_eg)

    # Black Total Material
    b_total_mg = b_mg + b_imb_score
    b_total_eg = b_eg + b_imb_score
    b_final = interpolate(b_total_mg, b_total_eg)

    return {'Material': {'White': w_final / DIV,
                        'Black': b_final / DIV,
                        'Diff': (w_final / DIV) - (b_final / DIV)}}



def calculate_position_stress_from_data(
    static_df: pd.DataFrame, 
    dynamic_lines: List[dict], # List of {score: cp, mate: int} from Client
    turn_color: bool
) -> float:
    """
    Refactored to use PRE-CALCULATED dynamic data from the client.
    """
    # 1. Handle MultiPV Data (Entropy)
    if not dynamic_lines: 
        return 0.0
        
    if len(dynamic_lines) == 1:
        search_criticality = 1.0 # Absolutely forced
    else:
        actual_pv = len(dynamic_lines)
        win_probs = []

        for line in dynamic_lines:
            # Client sends scores relative to the side to move
            # We normalize to raw Centipawns for the formula
            cp = line.get('score', 0)
            is_mate = line.get('is_mate', False)
            mate_val = line.get('mate', 0)

            if is_mate:
                cp = 10000 if mate_val > 0 else -10000
            
            # Sigmoid Win Probability
            w_p = 1 / (1 + math.pow(10, -cp / 400))
            win_probs.append(w_p)

        # Calculate Entropy
        w_sum = sum(win_probs)
        if w_sum < 1e-9:
            search_criticality = 0.0
        else:
            p_dist = [w / w_sum for w in win_probs]
            entropy = -sum(p * math.log(p) for p in p_dist if p > 0)
            
            max_entropy = math.log(actual_pv)
            search_criticality = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 1.0
            search_criticality = max(0.0, min(1.0, search_criticality))

    # 2. Determine Static Instability (Using your exact logic)
    color_str = "White" if turn_color == chess.WHITE else "Black"

    def get_raw_magnitude(feat):
        rows = static_df[(static_df['Feature'] == feat) & (static_df['Color'] == color_str)]
        if rows.empty: return 0.0
        return abs(rows['Value'].sum())

    danger = get_raw_magnitude('Danger')
    threats = (get_raw_magnitude('Threat Minor') + 
               get_raw_magnitude('Threat Rook') + 
               get_raw_magnitude('Hanging'))
    imbalance = get_raw_magnitude('Imbalance')
    stability = get_raw_magnitude('Connected') + get_raw_magnitude('Shelter Base')

    raw_instability = (danger * 1.5) + (threats * 1.0) + (imbalance * 0.5) - (stability * 0.5)
    
    # Sigmoid scaling for instability
    instability_score = 1 / (1 + math.exp(-(raw_instability - 100) / 100))

    final_stress = search_criticality * instability_score
    
    return round(final_stress, 4)


import chess
import math
from typing import Dict, List, Any

def classify_move_static(
    fen_before: str, 
    move_uci: str, 
    move_cp: float, 
    top_lines: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Classifies a chess move using pre-calculated engine data.
    
    Args:
        fen_before (str): FEN string of the position before the move.
        move_uci (str): The move played in UCI format (e.g., 'e2e4').
        move_cp (float): The centipawn score of the played move (from engine).
                         Positive values favor White, negative favor Black.
        top_lines (list): A list of dictionaries for the top engine moves, sorted by score.
                          Format: [{'uci': 'e2e4', 'cp': 35}, {'uci': 'd2d4', 'cp': 30}]
    
    Returns:
        dict: Classification results (category, accuracy, delta, etc.)
    """
    board = chess.Board(fen_before)
    move_obj = chess.Move.from_uci(move_uci)
    
    # 
    
    # --- 1. Material Count for Sacrifice Detection ---
    def get_material(b, color):
        # Simple material count (P=1, N/B=3, R=5, Q=9)
        vals = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
        return sum(len(b.pieces(pt, color)) * val for pt, val in vals.items())

    # We must track the material of the player who is currently moving
    mover = board.turn 
    mat_before = get_material(board, mover)
    
    board.push(move_obj)
    
    # After push, board.turn flips, so we look at the 'not board.turn' (the original mover)
    mat_after = get_material(board, mover) 
    board.pop() # Restore board state

    # Check if the mover has less material now (sacrificed pieces or unfavorable trade)
    material_dropped = (mat_before - mat_after) > 0

    # --- 2. Extract Scores & Win Probabilities ---
    def get_win_prob(cp):
        """Converts centipawns to win probability (0.0 to 1.0)."""
        # Handle mate scores (often +/- 10000 or similar large numbers)
        if cp > 1000: cp = 1000
        if cp < -1000: cp = -1000
        return 1 / (1 + math.pow(10, -cp / 400))

    # Adjust CP relative to the side to move (PovScore equivalent)
    # If it's Black's turn, a negative board score is good for Black.
    # We assume input CPs are "White-relative" (standard).
    turn_multiplier = 1 if mover == chess.WHITE else -1
    
    # Played Move Stats
    cp_played_pov = move_cp * turn_multiplier
    prob_played = get_win_prob(cp_played_pov)

    # Best Move Stats (from top_lines[0])
    if not top_lines:
        # Fallback if no lines provided
        best_move_uci = move_uci
        cp_best_pov = cp_played_pov
        prob_best = prob_played
        cp_2nd_pov = -10000 # irrelevant
        prob_2nd = 0.0
    else:
        best_move_data = top_lines[0]
        best_move_uci = best_move_data['uci']
        cp_best_pov = best_move_data['cp'] * turn_multiplier
        prob_best = get_win_prob(cp_best_pov)

        # Second Best Move Stats (for Criticality)
        if len(top_lines) > 1:
            cp_2nd_pov = top_lines[1]['cp'] * turn_multiplier
            prob_2nd = get_win_prob(cp_2nd_pov)
        else:
            prob_2nd = 0.0

    # --- 3. Classification Logic ---
    delta = prob_best - prob_played
    
    # Rounding delta to avoid floating point noise
    delta = max(0.0, round(delta, 4))
    
    category = "Unknown"

    # 

    if move_uci == best_move_uci:
        # It is the Best Move. Check for special tags.
        
        # Great Move Criteria:
        # 1. Position is winning/drawing (WinProb > 0.5)
        # 2. Significant drop to 2nd best move (Criticality > 0.3)
        is_critical = (prob_best - prob_2nd) > 0.3
        
        if material_dropped and cp_best_pov > 100: # Winning sacrifice
            category = "Brilliant"
        elif is_critical and prob_best > 0.6:
            category = "Great Move"
        else:
            category = "Best Move"
            
    elif delta < 0.02:
        category = "Excellent"
    elif delta < 0.1:
        category = "Good"
    elif delta < 0.25: # Slightly adjusted thresholds typical for accuracy
        category = "Inaccuracy"
    elif delta < 0.50: # Widened Mistake range
        category = "Mistake"
    else: # delta >= 0.50
        category = "Blunder"

    # --- 4. Accuracy Score (0-100) ---
    accuracy = 100 * math.exp(-2.0 * delta)

    return {
        "move": move_uci,
        "classification": category,
        "accuracy": round(accuracy, 1),
        "delta": round(delta, 4),
        "eval_best": cp_best_pov,
        "eval_played": cp_played_pov,
        "is_sacrifice": material_dropped
    }


import math
from typing import Dict

def analyze_time_management(
    time_left: float, 
    increment: float, 
    move_time: float, 
    move_number: int, 
    stress: float, 
    move_quality: str  # "Best Move", "Good", "Mistake", "Blunder", etc.
) -> Dict[str, any]:
    """
    Evaluates time management efficiency.
    
    Args:
        time_left (float): Seconds on clock BEFORE the move.
        increment (float): Seconds added per move.
        move_time (float): Seconds spent on this move.
        move_number (int): Current move number (1, 2, ...).
        stress (float): Position Stress Score (0.0 to 1.0).
        move_quality (str): Classification of the move played.
        
    Returns:
        Dict: Score (0.0-1.0), Classification (str), Target Time, and detailed stats.
    """
    
    # --- 1. Calculate Time Budget (The "Target") ---
    moves_horizon = max(20, 60 - move_number)
    base_budget = (time_left / moves_horizon) + increment
    
    # Stress Multiplier logic
    stress_multiplier = 0.2 + (2.5 * stress)
    target_time = base_budget * stress_multiplier
    
    # Cap target at 60% of remaining time to avoid absurd suggestions
    max_affordable = time_left * 0.6
    target_time = min(target_time, max_affordable)
    
    # --- 2. Calculate Efficiency Ratio ---
    ratio = move_time / max(1.0, target_time)
    
    # --- 3. Classification Logic ---
    label = "Optimal"
    score = 1.0
    
    is_bad_move = move_quality in ["Inaccuracy", "Mistake", "Blunder"]
    is_good_move = move_quality in ["Good", "Best Move", "Great Move", "Brilliant", "Excellent"]
    
    # A. Time Trouble Check
    in_time_trouble = time_left < 30 and increment < 5
    
    if in_time_trouble:
        if move_time < 1.0 and is_bad_move:
            label = "Panic"
            score = 0.4
        else:
            label = "Flagging"
            score = 0.8 # Survival mode is acceptable
            
    # B. The "Rushed" Category (Spent too little on a hard problem)
    elif ratio < 0.25:
        if stress > 0.6:
            if is_bad_move:
                label = "Reckless Rush"
                score = 0.2 # Terrible decision
            elif is_good_move:
                label = "Intuitive"
                score = 1.0 # Excellent intuition
            else:
                label = "Rushed"
                score = 0.6
        else:
            label = "Efficient" 
            score = 1.0

    # C. The "Overthinking" Category (Spent too much)
    elif ratio > 2.5:
        if stress < 0.3:
            label = "Overthinking"
            score = 0.5 # Waste of resources
        elif is_bad_move:
            label = "Paralysis"
            score = 0.3 # High cost, bad result
        else:
            label = "Deep Thought"
            score = 0.95

    # D. The "Optimal" Range
    else:
        label = "Normal"
        score = 1.0
        if is_bad_move:
            score = 0.8 # Penalty for outcome, but time usage was theoretically correct

    return {
        "score": score,
        "classification": label,
        "time_spent": round(move_time, 1),
        "target_time": round(target_time, 1),
        "efficiency_ratio": round(ratio, 2)
    }


import io
import re
import chess.pgn

def parse_pgn_data(pgn_string):
    """
    Parses a PGN string to extract time usage statistics in chronological order.
    Returns moves in UCI format (e.g., "e2e4").
    """
    
    pgn_io = io.StringIO(pgn_string)
    game = chess.pgn.read_game(pgn_io)
    
    if not game:
        return {"error": "Invalid PGN"}

    # --- Extract Time Control ---
    time_control = game.headers.get("TimeControl", "0")
    if "+" in time_control:
        start_time_str, increment_str = time_control.split("+")
        start_time = float(start_time_str)
        increment = float(increment_str)
    else:
        try:
            start_time = float(time_control)
        except ValueError:
            start_time = 0.0 
        increment = 0.0

    # --- Helper: HMS to Seconds ---
    def hms_to_seconds(hms_str):
        parts = hms_str.split(':')
        seconds = 0.0
        if len(parts) == 3: # H:MM:SS
            seconds += int(parts[0]) * 3600
            seconds += int(parts[1]) * 60
            seconds += float(parts[2])
        elif len(parts) == 2: # MM:SS
            seconds += int(parts[0]) * 60
            seconds += float(parts[1])
        return seconds

    # --- Tracking State ---
    white_clock = start_time
    black_clock = start_time
    
    ordered_moves = []
    
    node = game
    
    # --- Iterate Moves ---
    while node.variations:
        next_node = node.variation(0)
        
        # 1. Get Move in UCI (Universal Chess Interface) format
        # e.g., "e2e4" instead of "e4"
        move_played = next_node.move.uci()
        
        move_number = node.board().fullmove_number
        is_white_move = node.board().turn == chess.WHITE
        color = "White" if is_white_move else "Black"
        
        # 2. Extract Clock
        match = re.search(r'\[%clk\s+([\d:.]+)]', next_node.comment)
        
        if match:
            clk_str = match.group(1)
            current_clock = hms_to_seconds(clk_str)
            
            # 3. Calculate Time Spent
            if is_white_move:
                time_left = white_clock
                move_time = white_clock - current_clock + increment
                white_clock = current_clock
            else:
                time_left = black_clock
                move_time = black_clock - current_clock + increment
                black_clock = current_clock

            move_time = max(0.0, move_time)

            ordered_moves.append({
                "move_number": move_number,
                "color": color,
                "move": move_played, # Now strictly UCI
                "time_left": round(time_left, 2),
                "move_time": round(move_time, 2),
                "increment": increment
            })

        node = next_node

    return {
        "time_per_side_for_start": start_time,
        "time_increment": increment,
        "moves": ordered_moves
    }


def convert_numpy(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float64)):
        return float(obj)
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    return obj
    
def process_single_position(static_engine_path, i, data):
    position = data['positions'][i]
    fen = position['fen']

    # 1. Run Static Engine (Server Side)
    #engine = get_engine(static_engine_path)
    #raw_trace = engine.get_raw_eval(fen)
    cmd = f"ucinewgame\nposition fen {fen}\neval\nquit\n"
    proc = subprocess.run(
        static_engine_path, 
        input=cmd, 
        capture_output=True, 
        text=True, 
        check=True

    )
    raw_trace = proc.stdout
    df = parse_trace(raw_trace)
    
    
    pgn_data = parse_pgn_data(data['pgn'])
    static_features = extract_static_features(df)
    static_features.update(calculate_material_and_imbalance(fen))
    static_features['Criticality'] = calculate_position_stress_from_data(df, position['dynamic_data']['top_lines'], True)
    if len(position['dynamic_data']['top_lines']) >= 2:
        lines = [{'uci': line['uci'], 'cp': line['score']} for line in position['dynamic_data']['top_lines'][:2]]
    else:
        lines = [{'uci': position['dynamic_data']['top_lines'][0]['uci'], 'cp': position['dynamic_data']['top_lines'][0]['score']}, {'uci': position['dynamic_data']['top_lines'][0]['uci'], 'cp': position['dynamic_data']['top_lines'][0]['score']}]
    if i < len(data['positions']) - 1:
        static_features['Move'] = classify_move_static(fen, pgn_data['moves'][i]['move'], position['whiteEval'], lines)
    else:
        static_features['Move'] = None
    static_features['eval'] = position['whiteEval']
    if i < len(data['positions']) - 1:
        static_features['Time'] = analyze_time_management(pgn_data['moves'][i]['time_left'], pgn_data['time_increment'], pgn_data['moves'][i]['move_time'], pgn_data['moves'][i]['move_number'], static_features['Criticality'], static_features['Move']['classification'])
    else:
        static_features['Time'] = None
    return convert_numpy(static_features)




        