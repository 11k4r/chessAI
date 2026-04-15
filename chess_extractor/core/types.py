from enum import Enum
from typing import Dict
import chess

class PlayerColor(str, Enum):
    """Standardized string representations for players."""
    WHITE = "White"
    BLACK = "Black"

class GamePhase(str, Enum):
    """The master multiplier phases of the game."""
    OPENING = "Opening"
    MIDDLEGAME = "Middlegame"
    ENDGAME = "Endgame"

class BishopPairStatus(str, Enum):
    """Possession status of the bishop pair."""
    WHITE_PAIR = "White_Pair"
    BLACK_PAIR = "Black_Pair"
    NEITHER = "Neither"
    BOTH = "Both"

# Shared Engine/Human Piece Values
PIECE_VALUES: Dict[chess.PieceType, float] = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    # King has no standard float value in basic material counting
}

# Standardized string names for pieces
PIECE_NAMES: Dict[chess.PieceType, str] = {
    chess.PAWN: "Pawn",
    chess.KNIGHT: "Knight",
    chess.BISHOP: "Bishop",
    chess.ROOK: "Rook",
    chess.QUEEN: "Queen",
    chess.KING: "King"
}

def get_color_string(color: chess.Color) -> str:
    """Helper to convert python-chess boolean colors to standard strings."""
    return PlayerColor.WHITE.value if color == chess.WHITE else PlayerColor.BLACK.value