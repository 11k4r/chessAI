from dataclasses import dataclass
from typing import Dict, Set
import chess

from chess_extractor.core.types import PlayerColor

@dataclass
class PawnStructure:
    """
    Isolates the physical integrity and deformities of the pawn chains.
    """
    pawn_islands: Dict[str, int]
    passed_pawn_count: Dict[str, int]
    advanced_passed_pawns: Dict[str, int]     # Pawns on 5th, 6th, or 7th rank
    isolated_pawn_count: Dict[str, int]
    doubled_pawn_count: Dict[str, int]
    backward_pawn_count: Dict[str, int]
    pawn_chain_integrity: Dict[str, int]      # Number of pawns defended by friendly pawns
    blocked_pawn_count: Dict[str, int]        # Locked pawn rams

    @classmethod
    def extract(cls, board: chess.Board) -> 'PawnStructure':
        # Initialize dictionaries using the standard PlayerColor enum
        colors = [chess.WHITE, chess.BLACK]
        keys = [PlayerColor.WHITE.value, PlayerColor.BLACK.value]
        
        islands = {k: 0 for k in keys}
        passed = {k: 0 for k in keys}
        advanced_passed = {k: 0 for k in keys}
        isolated = {k: 0 for k in keys}
        doubled = {k: 0 for k in keys}
        backward = {k: 0 for k in keys}
        integrity = {k: 0 for k in keys}
        blocked = {k: 0 for k in keys}

        for color, key in zip(colors, keys):
            enemy_color = not color
            friendly_pawns = board.pieces(chess.PAWN, color)
            enemy_pawns = board.pieces(chess.PAWN, enemy_color)
            
            # Map files to lists of ranks where friendly pawns exist
            files_with_pawns: Dict[int, list[int]] = {f: [] for f in range(8)}
            for sq in friendly_pawns:
                files_with_pawns[chess.square_file(sq)].append(chess.square_rank(sq))

            enemy_files: Dict[int, list[int]] = {f: [] for f in range(8)}
            for sq in enemy_pawns:
                enemy_files[chess.square_file(sq)].append(chess.square_rank(sq))

            # --- 1. Pawn Islands ---
            # Count contiguous files that have at least one pawn
            island_count = 0
            in_island = False
            for f in range(8):
                if files_with_pawns[f]:
                    if not in_island:
                        island_count += 1
                        in_island = True
                else:
                    in_island = False
            islands[key] = island_count

            # Evaluate individual pawns
            for sq in friendly_pawns:
                file_idx = chess.square_file(sq)
                rank_idx = chess.square_rank(sq)
                
                # Forward direction depends on color (+1 for White, -1 for Black)
                forward_step = 1 if color == chess.WHITE else -1
                
                # --- 2. Doubled Pawns ---
                if len(files_with_pawns[file_idx]) > 1:
                    doubled[key] += 1

                # --- 3. Isolated Pawns ---
                has_adj_friendly = False
                for adj_f in [file_idx - 1, file_idx + 1]:
                    if 0 <= adj_f <= 7 and files_with_pawns[adj_f]:
                        has_adj_friendly = True
                        break
                if not has_adj_friendly:
                    isolated[key] += 1

                # --- 4. Passed Pawns & Advanced Passed Pawns ---
                is_passed = True
                for f in [file_idx - 1, file_idx, file_idx + 1]:
                    if 0 <= f <= 7:
                        for enemy_rank in enemy_files[f]:
                            # Enemy pawn is strictly "ahead" of this pawn
                            if (color == chess.WHITE and enemy_rank > rank_idx) or \
                               (color == chess.BLACK and enemy_rank < rank_idx):
                                is_passed = False
                                break
                    if not is_passed:
                        break
                
                if is_passed:
                    passed[key] += 1
                    # 5th, 6th, or 7th rank (0-indexed: White=4,5,6 / Black=3,2,1)
                    if (color == chess.WHITE and rank_idx in [4, 5, 6]) or \
                       (color == chess.BLACK and rank_idx in [3, 2, 1]):
                        advanced_passed[key] += 1

                # --- 5. Blocked Pawns (Pawn Rams) ---
                square_in_front = chess.square(file_idx, rank_idx + forward_step)
                if 0 <= square_in_front <= 63 and square_in_front in enemy_pawns:
                    blocked[key] += 1

                # --- 6. Pawn Chain Integrity (Defended by friendly pawn) ---
                is_defended = False
                for adj_f in [file_idx - 1, file_idx + 1]:
                    if 0 <= adj_f <= 7:
                        defending_rank = rank_idx - forward_step
                        if defending_rank in files_with_pawns[adj_f]:
                            is_defended = True
                            break
                if is_defended:
                    integrity[key] += 1

                # --- 7. Backward Pawns ---
                # A pawn is backward if it has friendly pawns on adjacent files, 
                # but they are ALL ahead of it, leaving it undefendable by pawns.
                if has_adj_friendly and not is_defended:
                    is_backward = True
                    for adj_f in [file_idx - 1, file_idx + 1]:
                        if 0 <= adj_f <= 7:
                            for adj_rank in files_with_pawns[adj_f]:
                                if (color == chess.WHITE and adj_rank <= rank_idx) or \
                                   (color == chess.BLACK and adj_rank >= rank_idx):
                                    is_backward = False # There's a pawn that is alongside or behind it
                                    break
                    if is_backward:
                        backward[key] += 1

        return cls(
            pawn_islands=islands,
            passed_pawn_count=passed,
            advanced_passed_pawns=advanced_passed,
            isolated_pawn_count=isolated,
            doubled_pawn_count=doubled,
            backward_pawn_count=backward,
            pawn_chain_integrity=integrity,
            blocked_pawn_count=blocked
        )