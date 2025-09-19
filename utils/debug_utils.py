"""
Debug utilities for MegaFS
"""

import torch
import time
from typing import Any, Dict, Optional
import os


class DebugLogger:
    """Debug logging utilities"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.logs = []
    
    def log(self, message: str, level: str = "INFO"):
        """Log a message with timestamp"""
        if not self.enabled:
            return
        
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        print(log_entry)
        self.logs.append(log_entry)
    
    def log_model_info(self, model: torch.nn.Module, name: str = "Model"):
        """Log model information"""
        if not self.enabled:
            return
        
        try:
            # Simple model info without parameter counting to avoid LazyModule issues
            device = next(model.parameters()).device if list(model.parameters()) else "unknown"
            self.log(f"{name} - Device: {device}, Type: {type(model).__name__}")
        except Exception as e:
            self.log(f"{name} - Info unavailable: {e}")
    
    def log_memory_usage(self, device: str = "cuda"):
        """Log GPU memory usage"""
        if not self.enabled or not torch.cuda.is_available():
            return
        
        allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(device) / 1024**3    # GB
        
        self.log(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    
    def log_weight_loading(self, filename: str, success: bool, keys: list = None):
        """Log weight loading results"""
        if not self.enabled:
            return
        
        status = "SUCCESS" if success else "FAILED"
        message = f"Weight loading {status}: {filename}"
        
        if keys:
            message += f" (keys: {keys})"
        
        self.log(message)
    
    def save_logs(self, filepath: str):
        """Save logs to file"""
        if not self.enabled or not self.logs:
            return
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write('\n'.join(self.logs))
            print(f"SUCCESS: Debug logs saved: {filepath}")
        except Exception as e:
            print(f"ERROR: Error saving logs: {e}")


class PerformanceProfiler:
    """Performance profiling utilities"""
    
    def __init__(self):
        self.timings = {}
    
    def start_timer(self, name: str):
        """Start timing an operation"""
        self.timings[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """End timing and return duration"""
        if name not in self.timings:
            return 0.0
        
        duration = time.time() - self.timings[name]
        del self.timings[name]
        return duration
    
    def log_timing(self, name: str, duration: float):
        """Log timing information"""
        print(f"TIMING: {name}: {duration:.3f}s")
    
    def profile_operation(self, name: str, operation, *args, **kwargs):
        """Profile an operation with timing"""
        self.start_timer(name)
        try:
            result = operation(*args, **kwargs)
            duration = self.end_timer(name)
            self.log_timing(name, duration)
            return result
        except Exception as e:
            self.end_timer(name)
            print(f"ERROR: Error in {name}: {e}")
            raise


def check_system_requirements():
    """Check system requirements and print status"""
    print("INFO: System Requirements Check:")
    
    # Python version
    import sys
    print(f"  Python: {sys.version}")
    
    # PyTorch version
    print(f"  PyTorch: {torch.__version__}")
    
    # CUDA availability
    if torch.cuda.is_available():
        print(f"  CUDA: Available (version {torch.version.cuda})")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        print("  CUDA: Not available")
    
    # Required directories
    required_dirs = ["weights", "models", "utils"]
    for dir_name in required_dirs:
        exists = os.path.exists(dir_name)
        status = "EXISTS" if exists else "MISSING"
        print(f"  {dir_name}/: {status}")
