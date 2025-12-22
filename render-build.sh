#!/usr/bin/env bash
# Exit on error
set -o errexit

# Install Python dependencies
pip install -r requirements.txt

echo "--- Engine Setup & Debug ---"

# Define paths
ENGINE_DIR="engines"
FINAL_PATH="$ENGINE_DIR/stockfish"

# Check if 'engines' folder exists
if [ ! -d "$ENGINE_DIR" ]; then
    echo "⚠️  Error: 'engines' directory missing!"
    exit 0
fi

# FIX: Deep Search for the Binary
# We look for any file that is executable or large, ignoring the 'stockfish' directory itself if it exists
echo "🔍 Searching for Stockfish binary in $ENGINE_DIR..."

# Find any file that contains "stockfish" in its name but is NOT a directory
# We limit depth to 5 to avoid infinite loops if symlinks exist
FOUND_BINARY=$(find "$ENGINE_DIR" -maxdepth 5 -type f -name "*stockfish*" | head -n 1)

if [ -n "$FOUND_BINARY" ]; then
    echo "   -> Found potential binary at: $FOUND_BINARY"
    
    # If the found file is NOT already at the final path, move it
    if [ "$FOUND_BINARY" != "$FINAL_PATH" ]; then
        echo "   -> Moving binary to $FINAL_PATH..."
        mv "$FOUND_BINARY" "${FINAL_PATH}_temp"
        
        # Clean up old directories if they were nested
        # Be careful not to delete the temp file we just made
        rm -rf "$ENGINE_DIR/stockfish" 2>/dev/null || true
        
        # Rename temp file to final location
        mv "${FINAL_PATH}_temp" "$FINAL_PATH"
        echo "   ✅ Binary successfully moved to $FINAL_PATH"
    else
        echo "   ✅ Binary is already in the correct location."
    fi
else
    echo "❌ Error: Could not find any Stockfish binary file in $ENGINE_DIR"
fi

# FIX 2: Apply Strong Permissions (755)
if [ -f "$FINAL_PATH" ]; then
    chmod 755 "$FINAL_PATH"
    echo "✅ Permissions set to 755 for $FINAL_PATH"
    
    # Debug: Print file details
    echo "--- Final File Details ---"
    ls -lh "$FINAL_PATH"
else
    echo "❌ Error: Final check failed. Stockfish binary missing at $FINAL_PATH"
fi
