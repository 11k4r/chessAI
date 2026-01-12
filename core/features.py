import pandas as pd
import math
import chess
from typing import Dict, List, Optional
from core.static_engine import get_engine
from core.parser import parse_trace

# ==========================================
# 1. PURE STATIC FEATURES (Your IP)
# ==========================================

def extract_static_features(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculates a comprehensive set of static evaluation features from a Stockfish 11 trace,
    EXCLUDING Material scores.
    
    Includes:
    1. Standard Stockfish Features: Mobility, King Safety, Pawn Structure, Space, Threats.
    2. Extended Strategic Features: Center Control, Activity, Harmony, Complexity, Weakness, Aggression, Integrity, Squeeze.
    
    All values are interpolated based on Game Phase/Scale Factor and normalized 
    so that 1.0 roughly equals the value of one Pawn (~213 internal units).
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
    
    # Game Phase: 128 = Start/Midgame, 0 = End of Endgame
    phase_row = df[df['Feature'] == 'Game Phase']
    phase = float(phase_row['Value'].iloc[0]) if not phase_row.empty else 128.0

    # Scale Factor: 64 is "Normal". Lower = drawish, Higher = winning. Applies to EG only.
    sf_row = df[df['Feature'] == 'Scale Factor']
    sf_raw = float(sf_row['Value'].iloc[0]) if not sf_row.empty else 64.0
    sf = sf_raw / 64.0

    # --- 2. Helper Functions ---

    def get_interpolated_score(mg: float, eg: float) -> float:
        """
        Stockfish Tapered Eval Formula:
        Score = (MG * Phase + EG * (128 - Phase) * ScaleFactor) / 128
        """
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

    results = {}

    # --- 3. Standard Stockfish Features Calculation (Without Material) ---
    
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
        'Space': [
            'Space'
        ],
        'Threats': [
            'Threat Minor', 'Threat Rook', 'Threat King', 'Hanging', 
            'Restricted', 'Safe Pawn', 'Pawn Push', 
            'Knight On Q', 'Slider On Q'
        ]
    }

    for category, atoms in sf_features_map.items():
        for color in ['White', 'Black']:
            mg_total = 0.0
            eg_total = 0.0
            for atom in atoms:
                rows = df[(df['Feature'] == atom) & (df['Color'] == color)]
                if not rows.empty:
                    mg_total += rows[rows['Phase'] == 'MG']['Value'].sum()
                    eg_total += rows[rows['Phase'] == 'EG']['Value'].sum()
            
            final_score = get_interpolated_score(mg_total, eg_total)
            results[f"{category}_{color}"] = final_score / DIV


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

    for color in ['White', 'Black']:
        opp_color = 'Black' if color == 'White' else 'White'
        
        # A. Center Control (Space + Pawn Occupancy Bonus)
        center_score = get_val('Space', color)
        pawn_feats = ['Isolated', 'Backward', 'Doubled', 'Connected', 'Weak Lever', 'Passed Rank']
        pawn_sqs = get_squares(pawn_feats, color)
        
        center_pawns = len([s for s in pawn_sqs if s in CENTER_SQUARES])
        ext_center_pawns = len([s for s in pawn_sqs if s in EXTENDED_CENTER])
        
        center_score += (center_pawns * W_CENTER_PAWN) + (ext_center_pawns * W_EXTENDED_PAWN)
        results[f'CenterControl_{color}'] = center_score / DIV

        # B. Activity
        activity = sum(get_val(f, color) for f in group_activity)
        results[f'Activity_{color}'] = activity / DIV

        # C. Harmony (Activity - Clumsiness)
        clumsiness = sum(abs(get_val(f, color)) for f in group_clumsiness)
        results[f'Harmony_{color}'] = (activity - clumsiness) / DIV

        # D. Complexity
        complexity = sum(abs(get_val(f, color)) for f in group_complexity)
        results[f'Complexity_{color}'] = complexity / DIV

        # E. Weakness (Sum of penalties)
        weakness = sum(abs(get_val(f, color)) for f in group_weakness)
        results[f'Weakness_{color}'] = weakness / DIV

        # F. Aggression (Threats + Initiative)
        aggression = sum(get_val(f, color) for f in group_aggression)
        winnable = get_val('Winnable Total', color)
        if winnable > 0: 
            aggression += winnable
        results[f'Aggression_{color}'] = aggression / DIV

        # G. Integrity (Structural Health)
        pos_integrity = sum(get_val(f, color) for f in group_integrity_pos)
        neg_integrity = sum(abs(get_val(f, color)) for f in group_integrity_neg)
        results[f'Integrity_{color}'] = (pos_integrity - neg_integrity) / DIV

        # H. Squeeze (Restriction + Mobility Diff)
        space = get_val('Space', color)
        restricted = get_val('Restricted', color)
        mob_diff = mobility_scores[color] - mobility_scores[opp_color]
        results[f'Squeeze_{color}'] = (space + restricted + mob_diff) / DIV

    return results


import chess

def calculate_material_and_imbalance(board: chess.Board):
    """
    Calculates Stockfish 11 Material and Imbalance scores from a python-chess Board.
    
    Returns a dictionary containing:
        - raw_mg, raw_eg: Raw material scores (White - Black)
        - imbalance: The net imbalance score (White - Black)
        - white_material_mg/eg: White's total material (Raw + Imbalance)
        - black_material_mg/eg: Black's total material (Raw + Imbalance)
        - phase: The calculated game phase (0=Endgame, 128=Midgame)
        - white_score, black_score: Final interpolated, normalized scores (1.0 = 1 Pawn)
    """
    
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

    return {
        "White_Material": w_final / DIV,
        "Black_Material": b_final / DIV,

    }
# ==========================================
# 2. HYBRID FEATURES (Refactored for Client-Data)
# ==========================================

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


def classify_move_from_data(
    move_uci: str,
    best_move_data: dict,   # {score, mate, uci}
    played_move_data: dict, # {score, mate, uci}
    second_best_data: Optional[dict] = None # Needed for "Great Move" detection
) -> Dict[str, any]:
    """
    Refactored to compare Client-side provided scores.
    """
    # Helper to convert CP/Mate dict to numeric score
    def get_cp(data):
        if data.get('is_mate'):
            return 10000 if data['mate'] > 0 else -10000
        return data['score']

    def get_win_prob(cp):
        return 1 / (1 + math.pow(10, -cp / 400))

    # 1. Extract Stats
    cp_best = get_cp(best_move_data)
    prob_best = get_win_prob(cp_best)

    cp_played = get_cp(played_move_data)
    prob_played = get_win_prob(cp_played)

    prob_2nd = 0.0
    if second_best_data:
        cp_2nd = get_cp(second_best_data)
        prob_2nd = get_win_prob(cp_2nd)

    # 2. Classification Logic
    delta = prob_best - prob_played
    category = "Unknown"
    
    best_move_uci = best_move_data.get('uci')

    if move_uci == best_move_uci:
        # Check Great Move Criteria (Simplified from your code)
        is_critical = (prob_best - prob_2nd) > 0.3
        
        # Note: Material drop logic requires Board history, usually handled in loop
        # For now, we assume standard Best Move unless critical
        if is_critical and prob_best > 0.6:
            category = "Great Move"
        else:
            category = "Best Move"
    
    elif delta < 0.02: category = "Excellent"
    elif delta < 0.1:  category = "Good"
    elif delta < 0.50: category = "Inaccuracy"
    elif delta < 1:    category = "Mistake"
    else:              category = "Blunder"

    accuracy = 100 * math.exp(-2.0 * delta)

    return {
        "move": move_uci,
        "classification": category,
        "accuracy": round(accuracy, 1),
        "delta": round(delta, 4),
        "eval_before": cp_best,
        "eval_after": cp_played
    }

# ==========================================
# 3. MAIN PROCESSOR (The Glue)
# ==========================================

def process_single_position(
    fen: str, 
    static_engine_path: str,
    dynamic_data: dict # Contains 'top_lines', 'played_move_eval', etc.
):
    """
    Runs the pipeline for one position.
    """
    board = chess.Board(fen)
    
    # 1. Run Static Engine (Server Side)
    engine = get_engine(static_engine_path)
    raw_trace = engine.get_raw_eval(fen)
    static_df = parse_trace(raw_trace)
    
    # 2. Extract Static Features
    static_feats = extract_static_features(static_df)
    material_feats = calculate_material_and_imbalance(board)
    
    # 3. Calculate Stress (Hybrid)
    # dynamic_data['top_lines'] is expected to be a list of {score, is_mate, mate, uci}
    stress = calculate_position_stress_from_data(
        static_df, 
        dynamic_data.get('top_lines', []), 
        board.turn
    )
    
    # 4. Merge
    return {
        **static_feats,
        **material_feats,
        "stress": stress
    }