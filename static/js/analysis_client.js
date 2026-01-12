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
            console.warn("⚠️ High-Performance Mode: OFF. Headers missing. Falling back to 1 Thread.");
            this.sendCommand('setoption name Threads value 1');
            this.sendCommand('setoption name Hash value 32');
        }

        // --- 4. ENGINE SETTINGS ---
        // SF 17 specific: 'Use NNUE' is usually on by default, but good to force if available
        this.sendCommand('setoption name Use NNUE value true');
        
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

        // Regex to extract depth, multipv, score, and pv (moves)
        // Example: info depth 16 ... multipv 1 score cp 50 ... pv e2e4 c7c5
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
        
        // Resolve the Promise with the collected data
        if (this.currentTask.resolve) {
            this.currentTask.resolve({
                fen: this.currentTask.fen,
                best_move: bestMove,
                top_lines: this.currentTask.results.filter(r => r) // Remove empty slots
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
        this.sendCommand(`go depth ${this.currentTask.targetDepth}`);
    }

    /**
     * BATCH PROCESSOR:
     * Loops through an entire game's FEN list, analyzes them locally,
     * and sends the big bundle to the Server.
     */
    async analyzeGameBatch(fenList, updateProgressCallback) {
        const batchData = [];
        
        // 1. Reset the engine ONCE at the start of the game
        this.sendCommand('ucinewgame');
        this.sendCommand('isready');
        // We give it a tiny delay to ensure it's ready (optional but safer)
        await new Promise(r => setTimeout(r, 50));

        for (let i = 0; i < fenList.length; i++) {
            const fen = fenList[i];
            
            if (updateProgressCallback) {
                updateProgressCallback(i + 1, fenList.length);
            }

            // 2. Run Analysis (Depth 12 is usually the sweet spot for browser batch)
            const dynamicResult = await this.evaluate(fen, 2);
            
            batchData.push({
                fen: fen,
                dynamic_data: dynamicResult
            });
        }

        console.log("Sending Batch to Server...", batchData);
        
        try {
            const response = await fetch('/api/analyze-batch', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ positions: batchData })
            });

            console.log("Server responded with status:", response.status);

            // 1. Get RAW TEXT first to debug
            const textData = await response.text();
            console.log("Raw Server Response (First 100 chars):", textData.substring(0, 100));

            // 2. Then parse it
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