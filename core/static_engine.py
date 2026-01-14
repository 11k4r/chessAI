import subprocess
import time
import os

class StaticEngine:
    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.process = None

    def start(self):
        """Starts the engine process."""
        if self.is_running():
            return
        if not os.path.exists(self.engine_path):
            raise FileNotFoundError(f"Engine not found at: {self.engine_path}")

        try:
            self.process = subprocess.Popen(
                [self.engine_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1
            )
            # Initialize
            self.send_command("ucinewgame")
            self.send_command("isready")
        except Exception as e:
            raise e

    def is_running(self):
        return self.process is not None and self.process.poll() is None

    def send_command(self, command):
        if not self.is_running():
            self.start()
        try:
            self.process.stdin.write(f"{command}\n")
            self.process.stdin.flush()
        except BrokenPipeError:
            self.process = None
            self.start()
            self.send_command(command)

    def get_raw_eval(self, fen):
        """
        Equivalent to your original 'run_eval', but uses the persistent process.
        Returns the raw string output of the 'eval' command.
        """
        self.send_command(f"position fen {fen}")
        self.send_command("eval")
        
        output_lines = []
        # We need a way to know when 'eval' is done.
        # Most engines end eval with "Final evaluation" or just stop sending.
        # Since 'eval' is instant, we can read until a specific marker or timeout.
        
        # NOTE: If your engine doesn't have a specific "end of eval" marker,
        # we might need to rely on a tiny timeout or specific keyword.
        # Does your output end with "Final evaluation: ..."?
        start = time.time()
        while True:
            line = self.process.stdout.readline()
            if not line: break
            output_lines.append(line)
            # STOP CONDITION: Adapt this to your engine's output!
            # Example: Stockfish 'eval' usually ends with "Final evaluation"
            if "Final evaluation" in line or "Total Evaluation" in line:
                break
                
            if time.time() - start > 0.5: # Safety break
                break
         
        return "".join(output_lines)

# Singleton helper
_engine = None
def get_engine(path):
    global _engine
    if _engine is None:
        _engine = StaticEngine(path)
        _engine.start()
    return _engine