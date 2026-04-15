from dataclasses import dataclass
from typing import Dict, List, Optional, Union

@dataclass
class MaterialBalance:
    total_material_difference: float          # e.g., +1.5 (White advantage)
    piece_by_piece_delta: Dict[str, str]      # e.g., {"White": "Rook", "Black": "Bishop, Knight"}
    bishop_pair_status: str                   # "White_Pair", "Black_Pair", "Neither", "Both"
    major_imbalances: str                     # e.g., "Queen vs Two Rooks", "None" (White first)
    minor_piece_ratio: str                    # e.g., "2N vs 2B" (White first)
    pawn_count_differential: int              # Raw count difference
    total_non_pawn_material: float            # Used for phase transition triggers
    sufficient_mating_material: Dict[str, bool] # {"White": True, "Black": False}

@dataclass
class PawnStructure:
    pawn_islands: Dict[str, int]
    passed_pawn_count: Dict[str, int]
    advanced_passed_pawns: Dict[str, int]     # Pawns on 5th, 6th, or 7th rank
    isolated_pawn_count: Dict[str, int]
    doubled_pawn_count: Dict[str, int]
    backward_pawn_count: Dict[str, int]
    pawn_chain_integrity: Dict[str, int]      # Number of mutually defended pawns
    blocked_pawn_count: Dict[str, int]        # Locked pawn rams

@dataclass
class KingSafety:
    pawn_shield_integrity: Dict[str, str]     # "Intact", "Compromised", "Destroyed"
    open_files_on_king: Dict[str, int]        # 0 to 3
    king_ring_attackers: Dict[str, int]       # Enemy pieces attacking the 3x3 grid
    king_ring_defenders: Dict[str, int]       # Friendly pieces defending the 3x3 grid
    king_xray_threats: Dict[str, int]
    escape_squares_luft: Dict[str, int]       # Number of safe legal king moves
    pawn_storm_proximity: Dict[str, int]      # Tracking depth/advancement of enemy pawns
    king_centralization_exposure: Dict[str, int] # Manhattan distance from center

@dataclass
class PieceActivity:
    safe_mobility_score: Dict[str, float]     # Safe legal moves available
    trapped_piece_count: Dict[str, int]       # Pieces with 0 or 1 safe moves
    bishop_scope: Dict[str, float]            # Average squares available per bishop
    advanced_rooks: Dict[str, int]            # Penetration to 7th/8th rank
    piece_centralization_index: Dict[str, float]
    development_status: Dict[str, int]        # Undeployed pieces count

@dataclass
class Coordination:
    outpost_synergy: Dict[str, int]
    connected_majors: Dict[str, int]
    battery_formations: Dict[str, int]
    hanging_piece_count: Dict[str, int]       # Pieces with 0 friendly defenders
    overprotection_score: Dict[str, float]
    piece_interference: Dict[str, int]        # Friendly pieces blocking optimal rays

@dataclass
class StaticTension:
    capture_volume: int                       # Total aggregate captures available
    pawn_tension: int                         # Active pawn-takes-pawn engagements
    en_prise_targets: Dict[str, int]          # Insufficiently defended pieces
    relative_pins: Dict[str, int]
    latent_discovered_attacks: Dict[str, int]
    non_king_xrays: Dict[str, int]
    active_forks_double_attacks: Dict[str, int]

@dataclass
class DecisionLandscape:
    viable_move_count: int                    # Moves within competitive threshold
    pv1_pv2_gap: float                        # Eval diff between best and 2nd best move
    blunder_risk_index: float                 # Percentage of moves causing > 2.0 eval drop
    evaluation_consistency: int               # How often top move changes during search
    drawishness_factor: int                   # Count of moves leading to dead draw evals

@dataclass
class GlobalContext:
    opening_name: str                         # ECO string
    game_phase_indicator: Union[float, str]   # "Opening", "Middlegame", "Endgame"
    castling_rights: Dict[str, bool]          # {"WK": True, "WQ": False, "BK": True, "BQ": True}
    halfmove_clock: int
    repetition_count: int
    color_to_move: str                        # "White" or "Black"
    endgame_type: Optional[str] = None        # e.g., "R+P vs R"
    en_passant_target: Optional[str] = None


@dataclass
class State:
    material: MaterialBalance
    structure: PawnStructure
    king_safety: KingSafety
    activity: PieceActivity
    coordination: Coordination
    tension: StaticTension
    landscape: DecisionLandscape
    context: GlobalContext