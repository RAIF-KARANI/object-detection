"""Quick sanity check: can PyTorch see and use the NVIDIA GPU?

Run with:  python check_gpu.py
"""
import torch


def main():
    """Print PyTorch/CUDA info and run one small calculation on the GPU."""
    print("PyTorch version :", torch.__version__)        # '+cu132' suffix = CUDA build
    print("CUDA build      :", torch.version.cuda)        # CUDA version PyTorch was compiled with (None = CPU-only build)
    print("CUDA available  :", torch.cuda.is_available())  # True only if the driver + GPU are usable

    if not torch.cuda.is_available():
        print("-> No usable GPU. Everything will still run on the CPU, just slower.")
        return

    print("GPU name        :", torch.cuda.get_device_name(0))  # device 0 = the first (only) NVIDIA GPU
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU memory      : {total_gb:.1f} GB")

    # Prove the GPU actually computes: multiply two random 1000x1000 matrices on it.
    a = torch.rand(1000, 1000, device="cuda")
    b = torch.rand(1000, 1000, device="cuda")
    c = a @ b                      # matrix multiplication, executed on the GPU
    torch.cuda.synchronize()       # GPU work is asynchronous; wait for it to finish
    print("Test matmul ran on:", c.device, "| result shape:", tuple(c.shape))


if __name__ == "__main__":
    main()
