import time

import ray
import torch


@ray.remote(num_gpus=1)
def gpu_heavy_job(job_id: int, iters: int = 200, size: int = 4096) -> dict:
    """Do a big matrix-mul loop on GPU so utilization is clearly visible."""
    import ray  # re-import inside worker

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        f"[Job {job_id}] device={device}, "
        f"ray.get_gpu_ids()={ray.get_gpu_ids()}, "
        f"torch.cuda.is_available()={torch.cuda.is_available()}",
        flush=True,
    )

    # Big matrices – this is what actually loads the GPU
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    torch.cuda.empty_cache()
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.time()

    # Heavy loop: repeated matmuls
    for i in range(iters):
        a = a @ b
        if device.type == "cuda" and (i + 1) % 10 == 0:
            # occasional sync so we measure real time and keep GPU busy
            torch.cuda.synchronize()

    if device.type == "cuda":
        torch.cuda.synchronize()

    end = time.time()
    elapsed = end - start

    return {
        "job_id": job_id,
        "elapsed_sec": elapsed,
        "size": size,
        "iters": iters,
        "mean": float(a.mean().item()),
    }


def main():
    ray.init()
    print("Ray resources:", ray.available_resources(), flush=True)

    # Launch a couple of heavy jobs; with a single GPU they’ll run sequentially,
    # but each one should keep the GPU close to 100% for a while.
    futures = [
        gpu_heavy_job.remote(job_id=1, iters=1000, size=8192),
        gpu_heavy_job.remote(job_id=2, iters=1000, size=8192),
    ]

    results = ray.get(futures)

    print("\nResults:")
    for r in results:
        print(
            f"Job {r['job_id']}: "
            f"{r['iters']} iters of {r['size']}x{r['size']} matmul "
            f"in {r['elapsed_sec']:.2f}s, mean={r['mean']:.4f}"
        )


if __name__ == "__main__":
    main()
