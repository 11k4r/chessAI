import pandas as pd
import numpy as np

class GameStatsCalculator:
    """
    2.x Game Analysis Metrics & Move Classification.
    Aggregates per-move data into player and game summaries.
    """
    
    @staticmethod
    def calculate(positions_data: list, result_str: str = "*") -> dict:
        if not positions_data:
            return {}
        
        # 1. Flatten Data
        # Convert list of dicts to DataFrame for easy column access
        # Nested keys become 'static.material.white', 'move_info.eval_delta', etc.
        df = pd.json_normalize(positions_data)
        
        # Helper: Calculate Accuracy (0-100) for a subset of moves
        def get_accuracy(subset):
            if subset.empty: return 0.0
            
            # Weighted sum of classifications
            weights = {
                "Brilliant": 100, "Best": 100, "Excellent": 95,
                "Good": 75, "Inaccuracy": 40, "Mistake": 10, "Blunder": 0
            }
            
            # Check if classification column exists
            if 'move_classification' not in subset.columns:
                return 0.0
                
            counts = subset['move_classification'].value_counts()
            total_score = sum(counts.get(cls, 0) * weight for cls, weight in weights.items())
            total_moves = len(subset)
            
            return round(total_score / total_moves, 1) if total_moves > 0 else 0.0

        stats = {"white": {}, "black": {}}
        
        # 2. Iterate Players to calculate specific stats
        for color_name in ["White", "Black"]:
            color_key = color_name.lower() # 'white' or 'black'
            
            # Filter moves made by this player
            # Note: 'move_info.turn' indicates who made the move resulting in that row
            if 'move_info.turn' not in df.columns:
                continue
                
            player_moves = df[df['move_info.turn'] == color_name]
            
            if player_moves.empty:
                continue

            # --- 2.2 Accuracy (Global) ---
            stats[color_key]['accuracy'] = get_accuracy(player_moves)
            
            # --- 2.3 - 2.5 Phase Accuracy ---
            # We group by the phase of the position
            if 'static.phase' in df.columns:
                for phase in ['Opening', 'Middlegame', 'Endgame']:
                    phase_moves = player_moves[player_moves['static.phase'] == phase]
                    stats[color_key][f'{phase.lower()}_accuracy'] = get_accuracy(phase_moves)

            # --- 2.6 Time Management ---
            # Requires %clk or TimeControl headers not present in basic PGNs
            stats[color_key]['time_management'] = 0.0

            # --- 2.7 Attack ---
            # Average of graph.attack score
            atk_col = f'graph.attack.{color_key}'
            if atk_col in df.columns:
                stats[color_key]['attack'] = round(player_moves[atk_col].mean(), 2)

            # --- 2.8 Defence ---
            def_col = f'graph.defence.{color_key}'
            if def_col in df.columns:
                stats[color_key]['defence'] = round(player_moves[def_col].mean(), 2)

            # --- 2.9 Aggression ---
            # Heuristic: Using Attack score for now. 
            # Could also be ratio of forward moves, but Attack is a good proxy.
            stats[color_key]['aggression'] = stats[color_key].get('attack', 0)

            # --- 2.10 Calculation ---
            # Heuristic: Average Complexity of positions this player faced
            if 'dynamic.complexity' in df.columns:
                stats[color_key]['calculation'] = round(player_moves['dynamic.complexity'].mean(), 2)

            # --- 2.11 Intuition ---
            # Heuristic: Accuracy in "Simple" positions (Low Complexity)
            if 'dynamic.complexity' in df.columns:
                simple_moves = player_moves[player_moves['dynamic.complexity'] < 2.0]
                stats[color_key]['intuition'] = get_accuracy(simple_moves)

            # --- 2.12 Strategic ---
            # Heuristic: Average Space + Center Control
            strat_score = 0
            if f'static.center_control.{color_key}' in df.columns:
                strat_score += player_moves[f'static.center_control.{color_key}'].mean()
            if f'static.space.{color_key}' in df.columns:
                strat_score += player_moves[f'static.space.{color_key}'].mean()
            stats[color_key]['strategic'] = round(strat_score, 2)

            # --- 2.13 Tactics ---
            # Heuristic: Accuracy in "Critical" positions (High Criticality)
            if 'dynamic.criticality' in df.columns:
                crit_moves = player_moves[player_moves['dynamic.criticality'] >= 5.0]
                stats[color_key]['tactics'] = get_accuracy(crit_moves)

            # --- 2.14 Move Class Distribution ---
            if 'move_classification' in player_moves.columns:
                stats[color_key]['move_distribution'] = player_moves['move_classification'].value_counts().to_dict()

            # --- 2.15 Opening Theory ---
            # Heuristic: % of "Best" moves in Opening phase
            if 'static.phase' in player_moves.columns and 'move_classification' in player_moves.columns:
                opening_moves = player_moves[player_moves['static.phase'] == 'Opening']
                if not opening_moves.empty:
                    best_count = len(opening_moves[opening_moves['move_classification'] == 'Best'])
                    stats[color_key]['opening_theory'] = round((best_count / len(opening_moves)) * 100, 1)
                else:
                    stats[color_key]['opening_theory'] = 0.0

            # --- 2.16 Key Moments ---
            # Count of Critical Positions faced
            if 'dynamic.criticality' in player_moves.columns:
                key_moments = len(player_moves[player_moves['dynamic.criticality'] > 7.0])
                stats[color_key]['key_moments_count'] = key_moments

            # --- 2.17 Volatility (Player Specific) ---
            # Std Dev of their Centipawn Loss (eval_delta)
            if 'move_info.eval_delta' in player_moves.columns:
                stats[color_key]['volatility'] = round(player_moves['move_info.eval_delta'].std(), 2)

            # --- 2.18 Resourcefulness ---
            # Accuracy when Eval is Bad (Defending a loss)
            # Eval is usually normalized. Let's assume bad is < -100 for current player.
            # We approximate by checking if the position result was bad for them? 
            # Or simplified: Average 'defence' score when under attack.
            # Let's use: Accuracy when 'graph.defence' is high (under pressure)
            if def_col in df.columns:
                high_def_moves = player_moves[player_moves[def_col] > 2.0]
                stats[color_key]['resourcefulness'] = get_accuracy(high_def_moves)

            # --- 2.19 Missed Opportunities ---
            # Blunders in winning positions (eval > 200)
            # Requires aligning previous eval. Simplified: Count of 'Blunder' moves.
            if 'move_classification' in player_moves.columns:
                stats[color_key]['missed_opportunities'] = int(len(player_moves[player_moves['move_classification'] == 'Blunder']))

            # --- 2.20 Pressure Handling ---
            # Accuracy when King Safety is compromised (> 5.0 implies unsafe)
            ks_col = f'graph.king_safety.{color_key}'
            if ks_col in df.columns:
                pressure_moves = player_moves[player_moves[ks_col] > 3.0]
                stats[color_key]['pressure_handling'] = get_accuracy(pressure_moves)

        # 3. Global Stats
        game_volatility = 0.0
        if 'dynamic.eval.value' in df.columns:
             # Clean data: convert mate dicts to float were handled in analyzer, 
             # but here we might see raw values if not fully processed.
             # Assuming 'dynamic.eval.value' is the numeric score.
             evals = pd.to_numeric(df['dynamic.eval.value'], errors='coerce').fillna(0).to_numpy()
             if len(evals) > 1:
                 game_volatility = float(np.std(np.diff(evals)))

        return {
            "result": result_str,
            "white": stats.get("white", {}),
            "black": stats.get("black", {}),
            "meta": {
                "game_volatility": round(game_volatility, 2),
                "total_moves": len(df)
            }
        }
        
        
        
        
       