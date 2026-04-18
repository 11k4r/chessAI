from dataclasses import dataclass
from typing import Dict
from enum import Enum
import chess

from chess_extractor.core.types import PlayerColor, PIECE_VALUES

_ROOK_DIRECTIONS   = [(0, 1), (0, -1), (1, 0), (-1, 0)]
_BISHOP_DIRECTIONS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
_ALL_DIRECTIONS    = _ROOK_DIRECTIONS + _BISHOP_DIRECTIONS

@dataclass
class StaticTension:
    """
    Measures the unresolved tactical friction, loaded engagements, 
    and pre-existing threats on the board before any move is made.
    """
    capture_volume: Dict[str, int]
    pawn_tension: int
    en_prise_targets: Dict[str, int]
    relative_pins: Dict[str, int]
    latent_discovered_attacks: Dict[str, int]
    non_king_xrays: Dict[str, int]
    active_forks_double_attacks: Dict[str, int]

    @classmethod
    def extract(cls, board: chess.Board) -> 'StaticTension':
        colors = [chess.WHITE, chess.BLACK]
        keys = [PlayerColor.WHITE.value, PlayerColor.BLACK.value]
        
        capture_volume = {k: 0 for k in keys}
        en_prise = {k: 0 for k in keys}
        relative_pins = {k: 0 for k in keys}
        latent_discovered = {k: 0 for k in keys}
        xrays = {k: 0 for k in keys}
        forks = {k: 0 for k in keys}
        
        # --- 1. Capture Volume (Per Player) ---
        original_turn = board.turn
        for color, key in zip(colors, keys):
            board.turn = color
            capture_volume[key] = len(list(board.generate_pseudo_legal_captures()))
        board.turn = original_turn

        # --- 2. Pawn Tension (Mutual Engagements) ---
        pawn_tension = 0
        white_pawns = board.pieces(chess.PAWN, chess.WHITE)
        black_pawns = board.pieces(chess.PAWN, chess.BLACK)
        for wp in white_pawns:
            attacks = board.attacks(wp)
            pawn_tension += len(attacks & black_pawns)

        for color, key in zip(colors, keys):
            enemy_color = not color
            
            friendly_pieces = chess.SquareSet(board.occupied_co[color])
            enemy_pieces = chess.SquareSet(board.occupied_co[enemy_color])
            
            # --- 3. En Prise Targets ---
            for sq in friendly_pieces:
                piece = board.piece_at(sq)
                if piece.piece_type == chess.KING:
                    continue 
                
                attackers = board.attackers(enemy_color, sq)
                if not attackers:
                    continue
                
                defenders = board.attackers(color, sq)
                piece_val = PIECE_VALUES.get(piece.piece_type, 0)
                
                is_en_prise = False
                
                if len(defenders) < len(attackers):
                    is_en_prise = True
                else:
                    min_attacker_val = min([PIECE_VALUES.get(board.piece_at(atk).piece_type, 0) for atk in attackers])
                    if min_attacker_val < piece_val:
                        is_en_prise = True
                        
                if is_en_prise:
                    en_prise[key] += 1

            # --- 4. Forks and Double Attacks ---
            for sq in friendly_pieces:
                piece = board.piece_at(sq)
                if not piece: continue
                
                attacks = board.attacks(sq)
                enemy_targets = attacks & enemy_pieces
                
                if len(enemy_targets) >= 2:
                    forks[key] += 1

            # --- 5, 6, 7. Ray Casting (Pins, Discovered, X-Rays) ---
            sliders = [sq for sq in friendly_pieces if board.piece_at(sq).piece_type in [chess.BISHOP, chess.ROOK, chess.QUEEN]]
            
            for sq in sliders:
                piece = board.piece_at(sq)
                file_idx = chess.square_file(sq)
                rank_idx = chess.square_rank(sq)
                
                directions = []
                if piece.piece_type in [chess.ROOK, chess.QUEEN]: directions.extend(_ROOK_DIRECTIONS)
                if piece.piece_type in [chess.BISHOP, chess.QUEEN]: directions.extend(_BISHOP_DIRECTIONS)
                
                for df, dr in directions:
                    pieces_encountered = []
                    f, r = file_idx + df, rank_idx + dr
                    
                    while 0 <= f <= 7 and 0 <= r <= 7:
                        target_sq = chess.square(f, r)
                        occupant = board.piece_at(target_sq)
                        if occupant:
                            pieces_encountered.append(occupant)
                        f += df
                        r += dr
                        
                        if len(pieces_encountered) == 3:
                            break 
                    
                    if len(pieces_encountered) >= 2:
                        p1, p2 = pieces_encountered[0], pieces_encountered[1]
                        
                        if p1.color == color and p2.color == enemy_color:
                            latent_discovered[key] += 1
                            
                        if p1.color == enemy_color and p2.color == enemy_color:
                            # FIX: Give the King an explicit high fallback value (100) so it registers as higher value
                            p1_val = PIECE_VALUES.get(p1.piece_type, 100 if p1.piece_type == chess.KING else 0)
                            p2_val = PIECE_VALUES.get(p2.piece_type, 100 if p2.piece_type == chess.KING else 0)
                            
                            # Check: Pawn is only pinned if attacked from the side (same rank)
                            is_valid_pawn_pin = True
                            if p1.piece_type == chess.PAWN and dr != 0:
                                is_valid_pawn_pin = False

                            # Include King in pins (the King's p2_val of 100 will easily pass this check)
                            if is_valid_pawn_pin and p2_val > p1_val:
                                relative_pins[key] += 1
                                
                            if p2.piece_type != chess.KING:
                                xrays[key] += 1

        return cls(
            capture_volume=capture_volume,
            pawn_tension=pawn_tension,
            en_prise_targets=en_prise,
            relative_pins=relative_pins,
            latent_discovered_attacks=latent_discovered,
            non_king_xrays=xrays,
            active_forks_double_attacks=forks
        )