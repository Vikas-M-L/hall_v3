"""Sample client process RSS; never infer remote GPU memory or API prices."""
import threading
import time


class MemorySampler:
    def __init__(self, interval=.05):
        self.interval = interval
        self.samples = []
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        try:
            import psutil
        except ImportError:
            return self
        process = psutil.Process()
        def sample():
            while not self.stop_event.is_set():
                self.samples.append(process.memory_info().rss)
                self.stop_event.wait(self.interval)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def finish(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=1)
        return {"sampled_peak_client_rss_bytes": max(self.samples) if self.samples else None,
                "sample_count": len(self.samples), "sample_interval_seconds": self.interval,
                "note": "sampled client RSS; may miss short peaks; excludes remote server memory"}
