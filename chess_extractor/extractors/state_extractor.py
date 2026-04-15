import chess
from dataclasses import dataclass
from typing import Dict

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

    @classmethod
    def from_board(cls, board: chess.Board) -> 'MaterialBalance':
        # Standard piece values for calculation
        values = {
            chess.PAWN: 1.0,
            chess.KNIGHT: 3.0,
            chess.BISHOP: 3.0,
            chess.ROOK: 5.0,
            chess.QUEEN: 9.0
        }
        names = {
            chess.PAWN: "Pawn",
            chess.KNIGHT: "Knight",
            chess.BISHOP: "Bishop",
            chess.ROOK: "Rook",
            chess.QUEEN: "Queen"
        }

        # 1. Count pieces for both sides
        w_counts = {p: len(board.pieces(p, chess.WHITE)) for p in values.keys()}
        b_counts = {p: len(board.pieces(p, chess.BLACK)) for p in values.keys()}

        # 2. Total Material & Differences
        w_total = sum(w_counts[p] * values[p] for p in values)
        b_total = sum(b_counts[p] * values[p] for p in values)
        total_difference = float(w_total - b_total)
        pawn_diff = w_counts[chess.PAWN] - b_counts[chess.PAWN]

        # Total Non-Pawn Material (Sum of both sides, excluding pawns)
        w_non_pawn = sum(w_counts[p] * values[p] for p in values if p != chess.PAWN)
        b_non_pawn = sum(b_counts[p] * values[p] for p in values if p != chess.PAWN)
        total_non_pawn = float(w_non_pawn + b_non_pawn)

        # 3. Piece-by-Piece Delta
        w_delta_list, b_delta_list = [], []
        
        # Order of importance for the delta string
        piece_order = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]
        
        for p in piece_order:
            diff = w_counts[p] - b_counts[p]
            if diff > 0:
                w_delta_list.extend([names[p]] * diff)
            elif diff < 0:
                b_delta_list.extend([names[p]] * abs(diff))

        piece_delta = {
            "White": ", ".join(w_delta_list) if w_delta_list else "None",
            "Black": ", ".join(b_delta_list) if b_delta_list else "None"
        }

        # 4. Bishop Pair Status
        w_pair = w_counts[chess.BISHOP] >= 2
        b_pair = b_counts[chess.BISHOP] >= 2
        if w_pair and b_pair:
            bishop_status = "Both"
        elif w_pair:
            bishop_status = "White_Pair"
        elif b_pair:
            bishop_status = "Black_Pair"
        else:
            bishop_status = "Neither"

        # 5. Minor Piece Ratio
        def format_minors(knights, bishops):
            if knights == 0 and bishops == 0: return "None"
            parts = []
            if knights > 0: parts.append(f"{knights}N" if knights > 1 else "N")
            if bishops > 0: parts.append(f"{bishops}B" if bishops > 1 else "B")
            return "+".join(parts)

        minor_ratio = f"{format_minors(w_counts[chess.KNIGHT], w_counts[chess.BISHOP])} vs {format_minors(b_counts[chess.KNIGHT], b_counts[chess.BISHOP])}"

        # 6. Major Imbalances
        imbalances = []
        w_surplus = {p: w_counts[p] - b_counts[p] for p in values if w_counts[p] > b_counts[p]}
        b_surplus = {p: b_counts[p] - w_counts[p] for p in values if b_counts[p] > w_counts[p]}

        # Check: Queen vs Two Rooks
        if w_surplus.get(chess.QUEEN, 0) >= 1 and b_surplus.get(chess.ROOK, 0) >= 2:
            imbalances.append("Queen vs Two Rooks")
        elif b_surplus.get(chess.QUEEN, 0) >= 1 and w_surplus.get(chess.ROOK, 0) >= 2:
            imbalances.append("Two Rooks vs Queen")

        # Check: Queen vs Three Minors
        b_minors_surplus = b_surplus.get(chess.KNIGHT, 0) + b_surplus.get(chess.BISHOP, 0)
        w_minors_surplus = w_surplus.get(chess.KNIGHT, 0) + w_surplus.get(chess.BISHOP, 0)
        
        if w_surplus.get(chess.QUEEN, 0) >= 1 and b_minors_surplus >= 3:
            imbalances.append("Queen vs Three Minor Pieces")
        elif b_surplus.get(chess.QUEEN, 0) >= 1 and w_minors_surplus >= 3:
            imbalances.append("Three Minor Pieces vs Queen")

        # Check: Exchange Imbalance (Rook vs Minor)
        if w_surplus.get(chess.ROOK, 0) >= 1 and (b_surplus.get(chess.KNIGHT, 0) >= 1 or b_surplus.get(chess.BISHOP, 0) >= 1):
            imbalances.append("Exchange Imbalance")
        elif b_surplus.get(chess.ROOK, 0) >= 1 and (w_surplus.get(chess.KNIGHT, 0) >= 1 or w_surplus.get(chess.BISHOP, 0) >= 1):
            imbalances.append("Exchange Imbalance")

        major_imbalance = ", ".join(imbalances) if imbalances else "None"

        # 7. Sufficient Mating Material
        # python-chess has_insufficient_material returns True if the player CANNOT mate
        mating_material = {
            "White": not board.has_insufficient_material(chess.WHITE),
            "Black": not board.has_insufficient_material(chess.BLACK)
        }

        return cls(
            total_material_difference=total_difference,
            piece_by_piece_delta=piece_delta,
            bishop_pair_status=bishop_status,
            major_imbalances=major_imbalance,
            minor_piece_ratio=minor_ratio,
            pawn_count_differential=pawn_diff,
            total_non_pawn_material=total_non_pawn,
            sufficient_mating_material=mating_material
        )