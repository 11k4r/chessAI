class AnalysisClient {
    constructor() {
        // Ensure you are pointing to your new Stockfish 17.1 files
        this.stockfish = new Worker('/static/wasm/stockfish-17.1-8e4d048.js'); 
        
        this.stockfish.onmessage = (event) => this.handleEngineMessage(event.data);
        
        this.resolveMap = {}; 
        this.isReady = false;
        
        // --- 1. INITIALIZE UCI ---
        this.sendCommand('uci');

        // --- 2. MAXIMIZE MULTI-THREADING ---
        // Check if high-performance mode is actually active
        if (window.crossOriginIsolated) {
            // Get logical cores (e.g., 8 on a modern laptop)
            const totalCores = navigator.hardwareConcurrency || 4;
            
            // RECOMMENDATION: Leave 1 core free for the Browser UI/Rendering
            // Otherwise, the browser might freeze while analyzing.
            const threadsToUse = Math.max(1, totalCores - 1);
            
            this.sendCommand(`setoption name Threads value ${threadsToUse}`);
            
            // --- 3. OPTIMIZE MEMORY (HASH) ---
            // Multi-threading requires a larger hash table to be effective.
            // 16MB (Default) is too small for 8 threads. 
            // 64MB or 128MB is safe for browsers; higher might crash mobile tabs.
            this.sendCommand('setoption name Hash value 128'); 
            
        } else {
            this.sendCommand('setoption name Threads value 1');
            this.sendCommand('setoption name Hash value 32');
        }

        // --- 4. ENGINE SETTINGS ---
        // SF 17 specific: 'Use NNUE' is usually on by default, but good to force if available
        this.sendCommand('setoption name Use NNUE value true');
        this.sendCommand('setoption name MultiPV value 5');
        this.sendCommand('isready');
    }

    sendCommand(cmd) {
        this.stockfish.postMessage(cmd);
    }

    handleEngineMessage(line) {
        // console.log("Engine:", line); // Debugging

        if (line === 'readyok') {
            this.isReady = true;
        }

        // Parse 'info' lines to extract scores and PVs
        if (line.startsWith('info depth')) {
            this.parseInfoLine(line);
        }

        // When search finishes for a move
        if (line.startsWith('bestmove')) {
            this.finishAnalysis(line);
        }
    }

    parseInfoLine(line) {
        if (!this.currentTask) return;


        const depthMatch = line.match(/depth (\d+)/);
        const multipvMatch = line.match(/multipv (\d+)/);
        const scoreMatch = line.match(/score (cp|mate) (-?\d+)/);
        const pvMatch = line.match(/ pv (.+)/);

        if (depthMatch && scoreMatch && pvMatch) {
            const depth = parseInt(depthMatch[1]);
            const mpv = multipvMatch ? parseInt(multipvMatch[1]) : 1;
            const type = scoreMatch[1];
            const value = parseInt(scoreMatch[2]);
            const moves = pvMatch[1];

            // Only update if we reached our target depth or higher
            if (depth >= this.currentTask.targetDepth) {
                this.currentTask.results[mpv - 1] = {
                    depth: depth,
                    is_mate: type === 'mate',
                    score: value, // Centipawns (or mate in X)
                    uci: moves.split(' ')[0], // Best move of this line
                    pv: moves
                };
            }
        }
    }

    finishAnalysis(bestMoveLine) {
        if (!this.currentTask) return;

        const bestMove = bestMoveLine.split(' ')[1];
        
        // 1. Determine Turn Color from FEN
        // FEN structure: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        // The second part ('w' or 'b') indicates the side to move.
        const fenParts = this.currentTask.fen.split(' ');
        const isBlack = (fenParts.length >= 2 && fenParts[1] === 'b');

        // 2. Prepare Lines (Filter & Normalize)
        const processedLines = this.currentTask.results
            .filter(r => r) // Remove empty slots
            .map(line => {
                // Create a shallow copy to avoid mutating raw state
                const newLine = { ...line };
                
                // NORMALIZE TO WHITE PERSPECTIVE
                // Stockfish returns scores relative to the side to move.
                // If it is Black's turn, we invert values so + is always White advantage.
                // This applies to both Centipawns and Mate scores.
                if (isBlack) {
                    newLine.score = -newLine.score;
                }
                return newLine;
            });

        // 3. Resolve with Normalized Data
        if (this.currentTask.resolve) {
            this.currentTask.resolve({
                fen: this.currentTask.fen,
                best_move: bestMove,
                top_lines: processedLines 
            });
        }
        
        this.currentTask = null;
        this.processQueue(); // Check if more FENs are waiting
    }

    // --- PUBLIC API ---

    /**
     * Queues a position for analysis.
     * Returns a Promise that resolves when analysis is complete.
     */
    evaluate(fen, depth = 16) {
        return new Promise((resolve, reject) => {
            this.queue = this.queue || [];
            this.queue.push({ fen, targetDepth: depth, resolve, reject, results: [] });
            this.processQueue();
        });
    }

    processQueue() {
        if (this.currentTask || !this.queue || this.queue.length === 0) return;
        
        // Start next task
        this.currentTask = this.queue.shift();
                
        this.sendCommand(`position fen ${this.currentTask.fen}`);
        this.sendCommand(`go depth ${10}`);
    }

    /**
     * BATCH PROCESSOR:
     * Loops through an entire game's FEN list, analyzes them locally,
     * and sends the big bundle to the Server.
     */
    async analyzeGameBatch(pgn, fenList, updateProgressCallback) {
        const batchData = [];
        
        // 1. Reset the engine
        this.sendCommand('ucinewgame');
        this.sendCommand('isready');
        await new Promise(r => setTimeout(r, 50));

        for (let i = 0; i < fenList.length; i++) {
            const fen = fenList[i];
            
            if (updateProgressCallback) {
                updateProgressCallback(i + 1, fenList.length);
            }

            const dynamicResult = await this.evaluate(fen, 10);
			
			let whiteEval = 0;
            if (dynamicResult.top_lines && dynamicResult.top_lines.length > 0) {
                const bestLine = dynamicResult.top_lines[0];
                whiteEval = bestLine.score;
            }
            
            batchData.push({
                fen: fen,
				whiteEval: whiteEval,
                dynamic_data: dynamicResult,
            });
        }

        
        try {
            // UPDATED: Send 'pgn' along with 'positions'
            const response = await fetch('/api/analyze-batch', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    pgn: pgn,          
                    positions: batchData 
                })
            });

            const textData = await response.text();
            const json = JSON.parse(textData);
            return json;

        } catch (err) {
            console.error("Batch Fetch Failed:", err);
            throw err;
        }
    }
}

// Global Instance
const chessEngine = new AnalysisClient();