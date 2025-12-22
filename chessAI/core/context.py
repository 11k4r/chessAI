import chess
import numpy as np

class AnalysisContext:
    """
    The Data Layer for Position Analysis.
    
    This class is instantiated once per position. It holds:
    1. The Bitboard representation (self.board) - For O(1) static checks.
    2. The Graph representation (self.*_matrix) - For O(N) interaction checks.
    3. The Engine Info (self.engine_info) - For dynamic evaluation.
    
    Architecture:
    - We pre-calculate the Adjacency Matrices (Attacks) in __init__.
    - This allows all Graph Metrics (Connectivity, Harmony, etc.) to run 
      instantly using Matrix Algebra instead of slow loops.
    """
    
    def __init__(self, board: chess.Board, engine_info: dict = None):
        self.board = board
        self.engine_info = engine_info or {}
        
        # --- GRAPH REPRESENTATION ---
        # 64x64 Adjacency Matrices
        # Rows = Attacker Square, Cols = Target Square
        # Value = 1 if attack exists, 0 otherwise
        self.white_attacks = np.zeros((64, 64), dtype=np.int8)
        self.black_attacks = np.zeros((64, 64), dtype=np.int8)
        
        # We build the graph immediately so metrics can consume it
        self._build_interaction_matrices()

    def _build_interaction_matrices(self):
        """
        Converts the Chess Board into Graph Adjacency Matrices.
        Optimized using python-chess bitboard iterators.
        """
        # Iterate over all squares to find who attacks whom
        for sq in chess.SQUARES:
            # Get attackers for this square (returns a Bitboard integer)
            w_attackers_bb = self.board.attackers(chess.WHITE, sq)
            b_attackers_bb = self.board.attackers(chess.BLACK, sq)
            
            # Update White Matrix
            # We iterate the set bits in the bitboard
            for attacker_sq in w_attackers_bb:
                self.white_attacks[attacker_sq, sq] = 1
                
            # Update Black Matrix
            for attacker_sq in b_attackers_bb:
                self.black_attacks[attacker_sq, sq] = 1

    @property
    def all_attacks(self):
        """Returns the combined attack graph."""
        return self.white_attacks + self.black_attacks