from dataclasses import dataclass
from typing import Dict, List, Set
import chess

from chess_extractor.core.types import PlayerColor


# Direction vectors for ray-tracing from the king
_ROOK_DIRECTIONS   = [(0, 1), (0, -1), (1, 0), (-1, 0)]
_BISHOP_DIRECTIONS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
_ALL_DIRECTIONS    = _ROOK_DIRECTIONS + _BISHOP_DIRECTIONS

_ROOK_DIR_SET   = set(map(tuple, _ROOK_DIRECTIONS))
_BISHOP_DIR_SET = set(map(tuple, _BISHOP_DIRECTIONS))

# Piece type sets for slider classification
_ROOK_SLIDERS   = {chess.ROOK, chess.QUEEN}
_BISHOP_SLIDERS = {chess.BISHOP, chess.QUEEN}

# The four central squares used for king-centralization measurement
_CENTER_SQUARES = [chess.D4, chess.D5, chess.E4, chess.E5]


@dataclass
class KingSafety:
    """
    Measures the localized defensive perimeter around both kings
    and their vulnerability to direct attack.
    """
    pawn_shield_integrity: Dict[str, int]
    """Count of intact shield pawns on the king's three flank files (0–3)."""

    open_files_on_king: Dict[str, int]
    """Open or semi-open files running through the king's flank (0–3)."""

    king_ring_attackers: Dict[str, int]
    """Distinct enemy pieces whose attack rays intersect the 3×3 king ring."""

    king_ring_defenders: Dict[str, int]
    """Distinct friendly pieces (excluding own king) guarding the 3×3 king ring."""

    king_xray_threats: Dict[str, int]
    """Enemy sliding pieces aligned with the king on a rank, file, or diagonal
    but currently blocked by at least one intervening piece."""

    escape_squares_luft: Dict[str, int]
    """Safe, unattacked squares the king can legally step to."""

    pawn_storm_proximity: Dict[str, int]
    """Rank-depth of the most advanced enemy pawn on the king's flank,
    measured from its starting rank (0 = still on start, higher = deeper storm)."""

    king_centralization_exposure: Dict[str, int]
    """Minimum Manhattan distance from the king to the four central squares
    (d4, d5, e4, e5). Low values signal danger in the middlegame."""

    @classmethod
    def extract(cls, board: chess.Board) -> 'KingSafety':
        colors = [chess.WHITE, chess.BLACK]
        keys   = [PlayerColor.WHITE.value, PlayerColor.BLACK.value]

        shield_integrity = {k: 0 for k in keys}
        open_files       = {k: 0 for k in keys}
        ring_attackers   = {k: 0 for k in keys}
        ring_defenders   = {k: 0 for k in keys}
        xray_threats     = {k: 0 for k in keys}
        escape_squares   = {k: 0 for k in keys}
        storm_proximity  = {k: 0 for k in keys}
        centralization   = {k: 0 for k in keys}

        for color, key in zip(colors, keys):
            enemy_color = not color

            king_sq = board.king(color)
            if king_sq is None:
                continue

            king_file = chess.square_file(king_sq)
            king_rank = chess.square_rank(king_sq)
            back_rank = 0 if color == chess.WHITE else 7

            # The three files forming the king's local flank (clamped to board)
            flank_files: List[int] = [
                f for f in (king_file - 1, king_file, king_file + 1)
                if 0 <= f <= 7
            ]

            friendly_pawns = board.pieces(chess.PAWN, color)
            enemy_pawns    = board.pieces(chess.PAWN, enemy_color)

            # ------------------------------------------------------------------
            # 1. Pawn Shield Integrity
            #    Count friendly pawns on flank files that are still within two
            #    ranks of the back rank (i.e. the shield has not been eroded).
            #    At most one pawn per file is counted; maximum score is 3.
            # ------------------------------------------------------------------
            shield_count = 0
            for f in flank_files:
                for sq in friendly_pawns:
                    if chess.square_file(sq) == f:
                        if abs(chess.square_rank(sq) - back_rank) <= 2:
                            shield_count += 1
                            break   # One shield pawn per file is enough
            shield_integrity[key] = shield_count

            # ------------------------------------------------------------------
            # 2. Open and Semi-Open Files on the King's Flank
            #    A file without any friendly pawn (open or semi-open) is a
            #    corridor that enemy heavy pieces can exploit.
            # ------------------------------------------------------------------
            open_file_count = 0
            for f in flank_files:
                has_friendly_pawn = any(chess.square_file(sq) == f for sq in friendly_pawns)
                if not has_friendly_pawn:
                    open_file_count += 1
            open_files[key] = open_file_count

            # ------------------------------------------------------------------
            # 3 & 4. King Ring Attackers and Defenders
            #    Build the 3×3 ring (up to 9 squares centred on the king),
            #    then collect the set of distinct attacking / defending pieces.
            # ------------------------------------------------------------------
            ring_squares: List[chess.Square] = [
                chess.square(king_file + df, king_rank + dr)
                for df in (-1, 0, 1)
                for dr in (-1, 0, 1)
                if 0 <= king_file + df <= 7 and 0 <= king_rank + dr <= 7
            ]

            attacker_set: Set[chess.Square] = set()
            defender_set: Set[chess.Square] = set()

            for sq in ring_squares:
                attacker_set.update(board.attackers(enemy_color, sq))

                for def_sq in board.attackers(color, sq):
                    piece = board.piece_at(def_sq)
                    # The king is not counted as its own ring defender
                    if piece and piece.piece_type != chess.KING:
                        defender_set.add(def_sq)

            ring_attackers[key] = len(attacker_set)
            ring_defenders[key] = len(defender_set)

            # ------------------------------------------------------------------
            # 5. King X-Ray Threats
            #    Cast a ray in each of the 8 directions from the king.
            #    When an enemy slider is found with at least one piece between
            #    it and the king, it constitutes an x-ray (pin / latent attack).
            #
            #    Algorithm:
            #      - Traverse squares in a direction, counting all pieces met.
            #      - If an enemy slider aligned to the current direction is
            #        reached and pieces_in_ray >= 1, record the x-ray and stop.
            #      - Stop unconditionally after three pieces (deeper x-rays are
            #        strategically irrelevant for localised safety scoring).
            # ------------------------------------------------------------------
            xray_count = 0
            for df, dr in _ALL_DIRECTIONS:
                is_rook_dir   = (df, dr) in _ROOK_DIR_SET
                is_bishop_dir = (df, dr) in _BISHOP_DIR_SET
                f, r          = king_file + df, king_rank + dr
                pieces_in_ray = 0

                while 0 <= f <= 7 and 0 <= r <= 7:
                    sq    = chess.square(f, r)
                    piece = board.piece_at(sq)

                    if piece is not None:
                        if piece.color == enemy_color and pieces_in_ray >= 1:
                            can_align = (
                                (is_rook_dir   and piece.piece_type in _ROOK_SLIDERS) or
                                (is_bishop_dir and piece.piece_type in _BISHOP_SLIDERS)
                            )
                            if can_align:
                                xray_count += 1
                                break

                        pieces_in_ray += 1
                        if pieces_in_ray >= 3:
                            break   # Practical depth limit

                    f += df
                    r += dr

            xray_threats[key] = xray_count

            # ------------------------------------------------------------------
            # 6. Escape Squares / Luft
            #    Iterate over the pseudo-legal king destination squares and keep
            #    only those that are (a) not occupied by a friendly piece and
            #    (b) not attacked by any enemy piece in the current position.
            # ------------------------------------------------------------------
            safe_moves = 0
            for target_sq in chess.SquareSet(chess.BB_KING_ATTACKS[king_sq]):
                occupant = board.piece_at(target_sq)
                if occupant and occupant.color == color:
                    continue   # Friendly piece blocks the square
                if board.is_attacked_by(enemy_color, target_sq):
                    continue   # Enemy covers the square
                safe_moves += 1
            escape_squares[key] = safe_moves

            # ------------------------------------------------------------------
            # 7. Pawn Storm Proximity
            #    Find the most advanced enemy pawn on the king's three flank
            #    files and report how many ranks it has traveled from its
            #    home rank (rank 1 for White, rank 6 for Black).
            #    Higher value = deeper, more dangerous storm.
            # ------------------------------------------------------------------
            max_storm_depth = 0
            for sq in enemy_pawns:
                if chess.square_file(sq) not in flank_files:
                    continue
                r = chess.square_rank(sq)
                # White storm pawns advance from rank 1 upward;
                # Black storm pawns advance from rank 6 downward.
                depth = (r - 1) if color == chess.BLACK else (6 - r)
                if depth > max_storm_depth:
                    max_storm_depth = depth
            storm_proximity[key] = max_storm_depth

            # ------------------------------------------------------------------
            # 8. King Centralization Exposure
            #    Minimum Manhattan distance from the king to any of the four
            #    central squares (d4, d5, e4, e5).
            # ------------------------------------------------------------------
            centralization[key] = min(
                abs(king_file - chess.square_file(c)) + abs(king_rank - chess.square_rank(c))
                for c in _CENTER_SQUARES
            )

        return cls(
            pawn_shield_integrity=shield_integrity,
            open_files_on_king=open_files,
            king_ring_attackers=ring_attackers,
            king_ring_defenders=ring_defenders,
            king_xray_threats=xray_threats,
            escape_squares_luft=escape_squares,
            pawn_storm_proximity=storm_proximity,
            king_centralization_exposure=centralization,
        )