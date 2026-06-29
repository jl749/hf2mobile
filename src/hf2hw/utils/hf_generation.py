import time
import transformers

class TokenSpeedStreamer(transformers.generation.streamers.BaseStreamer):
    def __init__(self):
        self.token_count = 0
        self.start_time: float | None = None

    def put(self, value):
        """Called every time the model generates a new token/batch."""
        if self.start_time is None:
            self.start_time = time.perf_counter()
            return

        num_tokens = value.numel()
        self.token_count += num_tokens
        
        elapsed = time.perf_counter() - self.start_time
        if elapsed > 0:
            current_speed = self.token_count / elapsed
            print(f"\rGenerated: {self.token_count} tokens | Speed: {current_speed:.2f} tok/sec", end="")

    def end(self):
        """Called when generation finishes."""
        if self.start_time is None:
            raise RuntimeError("self.start_time was never set!")
        elapsed = time.perf_counter() - self.start_time
        final_speed = self.token_count / elapsed
        print(f"\n\n✨ Done! Final Average Speed: {final_speed:.2f} tok/sec")


__all__ = ["TokenSpeedStreamer"]
