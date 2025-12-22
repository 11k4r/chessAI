#!/usr/bin/env bash
# Exit on error
set -o errexit

# Install Python dependencies
pip install -r requirements.txt

# Fix permissions for the Stockfish engine
# Ensure the engines directory exists first
if [ -d "engines" ]; then
    chmod +x engines/stockfish
    echo "✅ Permissions updated for engines/stockfish"
else
    echo "⚠️  Warning: 'engines' directory not found!"
fi
