from dataclasses import dataclass
from typing import Dict
import chess

from chess_extractor.core.types import PlayerColor

@dataclass
class CoordinationAndConnectivity:
    """
    Measures the synergy between pieces, tracking how they defend each other, 
    multiply offensive power, or restrict one another.
    """
    outpost_synergy: Dict[str, int]
    connected_majors: Dict[str, int]
    battery_formations: Dict[str, int]
    hanging_piece_count: Dict[str, int]
    overprotection_score: Dict[str, int]
    piece_interference: Dict[str, int]

    @classmethod
    def extract(cls, board: chess.Board) -> 'CoordinationAndConnectivity':
        colors = [chess.WHITE, chess.BLACK]
        keys = [PlayerColor.WHITE.value, PlayerColor.BLACK.value]
        
        outposts = {k: 0 for k in keys}
        connected_majors = {k: 0 for k in keys}
        batteries = {k: 0 for k in keys}
        hanging_pieces = {k: 0 for k in keys}
        overprotection = {k: 0 for k in keys}
        interference = {k: 0 for k in keys}

        for color, key in zip(colors, keys):
            enemy_color = not color
            friendly_pieces = board.occupied_co[color]
            
            # Pre-fetch piece lists for the current color
            pawns = board.pieces(chess.PAWN, color)
            knights = board.pieces(chess.KNIGHT, color)
            bishops = board.pieces(chess.BISHOP, color)
            rooks = board.pieces(chess.ROOK, color)
            queens = board.pieces(chess.QUEEN, color)
            
            all_friendly_squares = [sq for sq in chess.SQUARES if board.piece_at(sq) and board.piece_at(sq).color == color]

            # --- 1. Outpost Synergy ---
            # Minor pieces on rank 4-7 (White) or 0-3 (Black) defended by a friendly pawn
            advanced_ranks = [3, 4, 5, 6, 7] if color == chess.WHITE else [0, 1, 2, 3, 4]
            for sq in (knights | bishops):
                if chess.square_rank(sq) in advanced_ranks:
                    # Check if defended by friendly pawn
                    defenders = board.attackers(color, sq)
                    if any(board.piece_at(d).piece_type == chess.PAWN for d in defenders):
                        outposts[key] += 1

            # --- 2. Connected Majors & 3. Battery Formations ---
            majors = rooks | queens
            connected_count = 0
            battery_count = 0
            
            for sq in majors:
                piece_type = board.piece_at(sq).piece_type
                defended_by = board.attackers(color, sq)
                
                for d_sq in defended_by:
                    d_piece = board.piece_at(d_sq)
                    if not d_piece: continue
                    
                    # Connected Majors: Rooks/Queens defending each other on a straight line
                    if d_piece.piece_type in [chess.ROOK, chess.QUEEN]:
                        if chess.square_file(sq) == chess.square_file(d_sq) or chess.square_rank(sq) == chess.square_rank(d_sq):
                            connected_count += 1
                            # A straight-line connection of majors is also a battery
                            battery_count += 1
                            
                    # Battery Formations: Queen + Bishop on a diagonal
                    if (piece_type == chess.QUEEN and d_piece.piece_type == chess.BISHOP) or \
                       (piece_type == chess.BISHOP and d_piece.piece_type == chess.QUEEN):
                        battery_count += 1

            # Divide by 2 because A defends B and B defends A are counted twice for mutual connections
            connected_majors[key] = connected_count // 2
            batteries[key] = battery_count // 2

            # --- 4. Hanging Pieces & 5. Overprotection Score ---
            for sq in all_friendly_squares:
                pt = board.piece_at(sq).piece_type
                if pt == chess.KING:
                    continue # Kings are handled in King Safety

                friendly_defenders = len(board.attackers(color, sq))
                enemy_attackers = len(board.attackers(enemy_color, sq))

                # Hanging: 0 defenders (regardless of attack, latent vulnerability)
                if friendly_defenders == 0:
                    hanging_pieces[key] += 1
                
                # Overprotection: Defenders beyond what is mathematically necessary
                required_defenders = max(1, enemy_attackers)
                if friendly_defenders > required_defenders:
                    overprotection[key] += (friendly_defenders - required_defenders)

            # --- 6. Piece Interference ---
            # A friendly minor piece directly blocking a friendly Rook or Queen on a file
            for sq in (rooks | queens):
                file_idx = chess.square_file(sq)
                rank_idx = chess.square_rank(sq)
                
                # Check up and down the file for the *first* piece encountered
                for step in [1, -1]:
                    r = rank_idx + step
                    while 0 <= r <= 7:
                        target_sq = chess.square(file_idx, r)
                        occupant = board.piece_at(target_sq)
                        if occupant:
                            # If the first piece blocking the ray is a friendly minor piece, it's interference
                            if occupant.color == color and occupant.piece_type in [chess.KNIGHT, chess.BISHOP]:
                                interference[key] += 1
                            break # Ray stops at the first piece
                        r += step

        return cls(
            outpost_synergy=outposts,
            connected_majors=connected_majors,
            battery_formations=batteries,
            hanging_piece_count=hanging_pieces,
            overprotection_score=overprotection,
            piece_interference=interference
        )