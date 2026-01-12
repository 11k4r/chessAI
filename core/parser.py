import pandas as pd
import re

def parse_trace(output: str) -> pd.DataFrame:
    """
    Parses the raw 'eval' trace output into a Tidy DataFrame.
    Captures atomic features, values, and square metadata.
    """
    data = []

    # Regex for Game Phase
    phase_regex = re.compile(r"Game phase:\s+(\d+)")
    
    # Regex for Scale Factor
    sf_regex = re.compile(r"Scale Factor\s+\(Global, Global\),\s+\(([-\d]+),\s*([-\d]+)\)")
    
    # Updated Atom Regex
    atom_regex = re.compile(
        r"^([A-Za-z0-9 \-/]+)\s+"           # Group 1: Feature Name
        r"\(([-\d]+),\s*([-\d]+)\),\s+"     # Group 2, 3: White MG, EG
        r"\(([-\d]+),\s*([-\d]+)\)\s+"      # Group 4, 5: Black MG, EG
        r"\[(.*?)\],\s+\[(.*?)\]"           # Group 6, 7: White Metadata, Black Metadata
    )

    def parse_sq_list(sq_str):
        if not sq_str.strip(): return []
        return [s.strip() for s in sq_str.split(',') if s.strip()]

    for line in output.strip().splitlines():
        line = line.strip()
        
        # 1. Check Game Phase
        phase_match = phase_regex.search(line)
        if phase_match:
            data.append({
                'Feature': 'Game Phase', 
                'Color': 'Global', 
                'Phase': 'Global', 
                'Value': int(phase_match.group(1)),
                'metadata': []
            })
            continue

        # 2. Check Scale Factor
        sf_match = sf_regex.search(line)
        if sf_match:
            data.append({
                'Feature': 'Scale Factor', 
                'Color': 'Global', 
                'Phase': 'Global', 
                'Value': int(sf_match.group(1)),
                'metadata': []
            })
            continue

        # 3. Check Atomic Features
        match = atom_regex.match(line)
        if match:
            feat_name = match.group(1).strip()
            
            w_mg, w_eg, b_mg, b_eg = map(int, [
                match.group(2), match.group(3), 
                match.group(4), match.group(5)
            ])
            
            w_meta_list = parse_sq_list(match.group(6))
            b_meta_list = parse_sq_list(match.group(7))

            # White
            data.append({'Feature': feat_name, 'Color': 'White', 'Phase': 'MG', 'Value': w_mg, 'metadata': w_meta_list})
            data.append({'Feature': feat_name, 'Color': 'White', 'Phase': 'EG', 'Value': w_eg, 'metadata': w_meta_list})

            # Black
            data.append({'Feature': feat_name, 'Color': 'Black', 'Phase': 'MG', 'Value': b_mg, 'metadata': b_meta_list})
            data.append({'Feature': feat_name, 'Color': 'Black', 'Phase': 'EG', 'Value': b_eg, 'metadata': b_meta_list})

    return pd.DataFrame(data)