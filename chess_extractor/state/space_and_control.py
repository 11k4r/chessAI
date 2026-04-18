from dataclasses import dataclass
from typing import Dict
import chess

from chess_extractor.core.types import PlayerColor

# Standard 4 central squares
_CENTER_SQUARES = [chess.D4, chess.D5, chess.E4, chess.E5]

# Expanded center (c4, c5, f4, f5 + d4, d5, e4, e5)
_EXPANDED_CENTER = [
    chess.C4, chess.C5, chess.D4, chess.D5, 
    chess.E4, chess.E5, chess.F4, chess.F5
]

# Kingside (f, g, h) and Queenside (a, b, c) files
_KINGSIDE_FILES = [5, 6, 7]
_QUEENSIDE_FILES = [0, 1, 2]

@dataclass
class SpaceAndControl:
    """
    Measures the amount of physical territory dictated by each player,
    central dominance, and spatial advantages.
    """
    center_control_index: Dict[str, int]
    pawn_center_presence: Dict[str, int]
    enemy_territory_control: Dict[str, int]
    space_advantage_metric: Dict[str, int]
    flank_domination_balance: Dict[str, str]
    contested_squares_density: int  # Global metric

    @classmethod
    def extract(cls, board: chess.Board) -> 'SpaceAndControl':
        colors = [chess.WHITE, chess.BLACK]
        keys = [PlayerColor.WHITE.value, PlayerColor.BLACK.value]
        
        center_control = {k: 0 for k in keys}
        pawn_center = {k: 0 for k in keys}
        enemy_territory = {k: 0 for k in keys}
        space_advantage = {k: 0 for k in keys}
        flank_balance = {k: "" for k in keys}
        contested_density = 0

        # --- 1. Global Contested Squares ---
        for sq in chess.SQUARES:
            if board.is_attacked_by(chess.WHITE, sq) and board.is_attacked_by(chess.BLACK, sq):
                contested_density += 1

        for color, key in zip(colors, keys):
            enemy_color = not color
            
            # --- 2. Center Control Index ---
            # Sum of all friendly attacks on the 4 central squares
            for sq in _CENTER_SQUARES:
                attackers = board.attackers(color, sq)
                center_control[key] += len(attackers)

            # --- 3. Pawn Center Presence ---
            # Pawns occupying the expanded center
            for sq in _EXPANDED_CENTER:
                piece = board.piece_at(sq)
                if piece and piece.piece_type == chess.PAWN and piece.color == color:
                    pawn_center[key] += 1

            # --- 4. Enemy Territory Control ---
            # Ranks 5-8 (idx 4-7) for White; Ranks 1-4 (idx 0-3) for Black
            enemy_ranks = [4, 5, 6, 7] if color == chess.WHITE else [0, 1, 2, 3]
            territory_count = 0
            for r in enemy_ranks:
                for f in range(8):
                    sq = chess.square(f, r)
                    if board.is_attacked_by(color, sq):
                        territory_count += 1
            enemy_territory[key] = territory_count

            # --- 5. Space Advantage Metric ---
            # Number of "safe" squares behind the most advanced friendly pawn on each file
            safe_squares_behind_pawns = 0
            friendly_pawns = board.pieces(chess.PAWN, color)
            
            # Find the most advanced pawn per file
            advanced_pawn_ranks = {}
            for sq in friendly_pawns:
                f, r = chess.square_file(sq), chess.square_rank(sq)
                if f not in advanced_pawn_ranks:
                    advanced_pawn_ranks[f] = r
                else:
                    if color == chess.WHITE and r > advanced_pawn_ranks[f]:
                        advanced_pawn_ranks[f] = r
                    elif color == chess.BLACK and r < advanced_pawn_ranks[f]:
                        advanced_pawn_ranks[f] = r

            # Count safe squares behind those pawns
            for f, advanced_r in advanced_pawn_ranks.items():
                ranks_behind = range(0, advanced_r) if color == chess.WHITE else range(advanced_r + 1, 8)
                for r in ranks_behind:
                    sq = chess.square(f, r)
                    # A square is "safe" if it is not attacked by the enemy
                    if not board.is_attacked_by(enemy_color, sq):
                        safe_squares_behind_pawns += 1
            space_advantage[key] = safe_squares_behind_pawns

            # --- 6. Flank Domination Balance ---
            # Compare attacks on Kingside vs Queenside
            ks_attacks = sum(1 for r in range(8) for f in _KINGSIDE_FILES if board.is_attacked_by(color, chess.square(f, r)))
            qs_attacks = sum(1 for r in range(8) for f in _QUEENSIDE_FILES if board.is_attacked_by(color, chess.square(f, r)))
            
            diff = ks_attacks - qs_attacks
            if diff > 0:
                flank_balance[key] = f"Kingside +{diff}"
            elif diff < 0:
                flank_balance[key] = f"Queenside +{abs(diff)}"
            else:
                flank_balance[key] = "Balanced (0)"

        return cls(
            center_control_index=center_control,
            pawn_center_presence=pawn_center,
            enemy_territory_control=enemy_territory,
            space_advantage_metric=space_advantage,
            flank_domination_balance=flank_balance,
            contested_squares_density=contested_density
        )