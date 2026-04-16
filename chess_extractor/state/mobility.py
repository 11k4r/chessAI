from dataclasses import dataclass
from typing import Dict
import chess

from chess_extractor.core.types import PlayerColor, PIECE_VALUES

_CENTER_SQUARES = [chess.D4, chess.D5, chess.E4, chess.E5]

_STARTING_SQUARES = {
    chess.WHITE: {
        chess.KNIGHT: [chess.B1, chess.G1],
        chess.BISHOP: [chess.C1, chess.F1],
        chess.ROOK: [chess.A1, chess.H1],
        chess.QUEEN: [chess.D1]
    },
    chess.BLACK: {
        chess.KNIGHT: [chess.B8, chess.G8],
        chess.BISHOP: [chess.C8, chess.F8],
        chess.ROOK: [chess.A8, chess.H8],
        chess.QUEEN: [chess.D8]
    }
}

@dataclass
class Mobility:
    """
    Measures the range, placement quality, and movement freedom of pieces
    on an individual basis.
    """
    safe_mobility_score: Dict[str, float]
    trapped_piece_count: Dict[str, int]
    bishop_scope: Dict[str, float]
    advanced_rooks: Dict[str, int]
    piece_centralization_index: Dict[str, float]
    development_status: Dict[str, int]

    @classmethod
    def extract(cls, board: chess.Board) -> 'Mobility':
        colors = [chess.WHITE, chess.BLACK]
        keys = [PlayerColor.WHITE.value, PlayerColor.BLACK.value]
        
        safe_mobility = {k: 0.0 for k in keys}
        trapped_pieces = {k: 0 for k in keys}
        bishop_scope = {k: 0.0 for k in keys}
        advanced_rooks = {k: 0 for k in keys}
        centralization = {k: 0.0 for k in keys}
        development = {k: 0 for k in keys}

        for color, key in zip(colors, keys):
            enemy_color = not color
            
            # ------------------------------------------------------------------
            # 1. Development Status
            # Count of minor and major pieces that remain on their starting squares.
            # ------------------------------------------------------------------
            undeployed_count = 0
            for pt in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
                for sq in _STARTING_SQUARES[color][pt]:
                    piece = board.piece_at(sq)
                    if piece and piece.color == color and piece.piece_type == pt:
                        undeployed_count += 1
            development[key] = undeployed_count

            # ------------------------------------------------------------------
            # 2. Advanced Rooks
            # Rooks that have penetrated to the 7th/8th rank (2nd/1st for Black).
            # ------------------------------------------------------------------
            rooks = board.pieces(chess.ROOK, color)
            adv_rooks_count = 0
            for r_sq in rooks:
                rank = chess.square_rank(r_sq)
                # 0-indexed ranks: 6, 7 are 7th/8th; 0, 1 are 1st/2nd
                if color == chess.WHITE and rank in [6, 7]:
                    adv_rooks_count += 1
                elif color == chess.BLACK and rank in [0, 1]:
                    adv_rooks_count += 1
            advanced_rooks[key] = adv_rooks_count

            # ------------------------------------------------------------------
            # 3. Piece Centralization Index
            # Aggregate Manhattan distance of minor/major pieces to the 4 central squares.
            # ------------------------------------------------------------------
            cent_dist = 0
            active_major_minor = []
            for pt in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
                pieces = board.pieces(pt, color)
                for sq in pieces:
                    active_major_minor.append((sq, pt))
                    min_dist = min(
                        abs(chess.square_file(sq) - chess.square_file(c)) + 
                        abs(chess.square_rank(sq) - chess.square_rank(c))
                        for c in _CENTER_SQUARES
                    )
                    cent_dist += min_dist
            centralization[key] = float(cent_dist)

            # ------------------------------------------------------------------
            # 4. Bishop Scope
            # The average number of unobstructed squares available to bishops.
            # ------------------------------------------------------------------
            bishops = board.pieces(chess.BISHOP, color)
            total_b_scope = 0
            for b_sq in bishops:
                # board.attacks gives the rays stemming from the piece until a blocker
                scope = len(board.attacks(b_sq))
                total_b_scope += scope
            
            if len(bishops) > 0:
                bishop_scope[key] = float(total_b_scope) / len(bishops)
            else:
                bishop_scope[key] = 0.0

            # ------------------------------------------------------------------
            # 5 & 6. Safe Mobility Score & Trapped Piece Count
            # Track legal moves per piece. If a piece lands on a square controlled
            # by a lower-value enemy piece, the move is excluded from "safe" counts.
            # ------------------------------------------------------------------
            # Use sets to track unique destination squares (filters out promotion duplicates)
            piece_safe_destinations = {sq: set() for sq, _ in active_major_minor}
            safe_moves_set = set() # Tracks unique (from_sq, to_sq) pairs globally

            # Temporarily simulate the turn to get valid legal moves for this color
            original_turn = board.turn
            board.turn = color
            legal_moves = list(board.legal_moves)
            board.turn = original_turn

            for move in legal_moves:
                from_sq = move.from_square
                to_sq = move.to_square
                
                # If we've already evaluated this physical square transition, skip it
                if (from_sq, to_sq) in safe_moves_set:
                    continue
                
                piece = board.piece_at(from_sq)
                if piece is None: 
                    continue

                is_safe = True
                
                # Check for lower-value enemy domination
                if piece.piece_type not in [chess.PAWN, chess.KING]:
                    enemy_attackers = board.attackers(enemy_color, to_sq)
                    piece_val = PIECE_VALUES.get(piece.piece_type, 0)
                    
                    for atk_sq in enemy_attackers:
                        atk_piece = board.piece_at(atk_sq)
                        if atk_piece:
                            # If attacked by a lower value piece, it's not a "safe" mobility square
                            atk_val = PIECE_VALUES.get(atk_piece.piece_type, 0)
                            if atk_val < piece_val:
                                is_safe = False
                                break
                
                if is_safe:
                    safe_moves_set.add((from_sq, to_sq))
                    if from_sq in piece_safe_destinations:
                        piece_safe_destinations[from_sq].add(to_sq)

            safe_mobility[key] = float(len(safe_moves_set))

            # Pieces with 0 or 1 unique safe destination are considered trapped
            trapped_count = sum(1 for dests in piece_safe_destinations.values() if len(dests) <= 1)
            trapped_pieces[key] = trapped_count

        return cls(
            safe_mobility_score=safe_mobility,
            trapped_piece_count=trapped_pieces,
            bishop_scope=bishop_scope,
            advanced_rooks=advanced_rooks,
            piece_centralization_index=centralization,
            development_status=development
        )