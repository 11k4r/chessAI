from dataclasses import dataclass
from typing import Dict, Any
import chess
from chess_extractor.core.types import PlayerColor, GamePhase, PIECE_VALUES

@dataclass
class GlobalContext:
    opening_name: str
    endgame_type: str
    game_phase_indicator: str
    castling_rights: Dict[str, bool]
    halfmove_clock: int
    repetition_count: int
    en_passant_target: str
    color_to_move: str

    @classmethod
    def extract(cls, board: chess.Board, current_fullmove: int, eco_db: Dict[str, Any]) -> 'GlobalContext':
        # 1. Opening Name Extraction
        # STRIP DOWN TO 3 PARTS (Pieces, Turn, Castling) to ensure perfect matching
        base_fen = " ".join(board.fen().split(" ")[:3])
        opening_info = eco_db.get(base_fen, None)
        opening_name = opening_info.get("name", "Unknown") if opening_info else "Unknown"

        # 2. Endgame Type Classification
        def get_material_string(color: chess.Color) -> str:
            pieces = []
            piece_types = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]
            piece_chars = {chess.QUEEN: "Q", chess.ROOK: "R", chess.BISHOP: "B", chess.KNIGHT: "N", chess.PAWN: "P"}
            
            for pt in piece_types:
                count = len(board.pieces(pt, color))
                if count == 1:
                    pieces.append(piece_chars[pt])
                elif count > 1:
                    pieces.append(f"{count}{piece_chars[pt]}")
            
            return "K+" + "+".join(pieces) if pieces else "K"

        w_endgame_str = get_material_string(chess.WHITE)
        b_endgame_str = get_material_string(chess.BLACK)
        endgame_type = f"{w_endgame_str} vs {b_endgame_str}"

        if "B" in w_endgame_str and "B" in b_endgame_str and "Q" not in endgame_type and "R" not in endgame_type:
            w_bishops = board.pieces(chess.BISHOP, chess.WHITE)
            b_bishops = board.pieces(chess.BISHOP, chess.BLACK)
            if len(w_bishops) == 1 and len(b_bishops) == 1:
                w_sq = w_bishops.pop()
                b_sq = b_bishops.pop()
                w_sq_color = (chess.square_rank(w_sq) + chess.square_file(w_sq)) % 2
                b_sq_color = (chess.square_rank(b_sq) + chess.square_file(b_sq)) % 2
                if w_sq_color != b_sq_color:
                    endgame_type += " (Opposite Colored)"

        # 3. Game Phase Determination
        non_pawn_material = 0
        for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            w_count = len(board.pieces(piece_type, chess.WHITE))
            b_count = len(board.pieces(piece_type, chess.BLACK))
            val = PIECE_VALUES.get(piece_type, {chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}[piece_type])
            non_pawn_material += (w_count + b_count) * val

        ENDGAME_MATERIAL_THRESHOLD = 15.0 
        is_in_opening_book = opening_name != "Unknown"
        
        if is_in_opening_book:
            game_phase = GamePhase.OPENING.value
        elif non_pawn_material <= ENDGAME_MATERIAL_THRESHOLD:
            game_phase = GamePhase.ENDGAME.value
        elif current_fullmove <= 15:
            game_phase = GamePhase.OPENING.value
        else:
            game_phase = GamePhase.MIDDLEGAME.value

        # 4-7. Remaining Context Features
        castling_rights = {
            "WK": board.has_kingside_castling_rights(chess.WHITE),
            "WQ": board.has_queenside_castling_rights(chess.WHITE),
            "BK": board.has_kingside_castling_rights(chess.BLACK),
            "BQ": board.has_queenside_castling_rights(chess.BLACK)
        }

        transposition_key = board._transposition_key()
        repetition_count = board._transposition_table.count(transposition_key) if hasattr(board, '_transposition_table') else 1

        en_passant_target = chess.square_name(board.ep_square) if board.ep_square is not None else "None"
        color_to_move = PlayerColor.WHITE.value if board.turn == chess.WHITE else PlayerColor.BLACK.value

        return cls(
            opening_name=opening_name,
            endgame_type=endgame_type,
            game_phase_indicator=game_phase,
            castling_rights=castling_rights,
            halfmove_clock=board.halfmove_clock,
            repetition_count=max(1, repetition_count),
            en_passant_target=en_passant_target,
            color_to_move=color_to_move
        )