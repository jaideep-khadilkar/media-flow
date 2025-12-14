import ray
import time
import os


def initialize_ray():
    ray.init(
        address="ray://ray-head:10001",
        ignore_reinit_error=True,
    )
    print("Connected to existing Ray cluster.")


@ray.remote
def heavy_pure_python_task(task_id: int) -> float:
    """
    A CPU-intensive task using only pure Python (no NumPy or external libs).
    Performs a large number of floating-point power operations to keep the CPU busy
    for several seconds while being GIL-bound.
    """
    print(f"Starting heavy task {task_id} on worker PID {os.getpid()}")

    total = 0.0
    iterations = 100_000_000  # Adjust higher/lower depending on your CPU

    start_time = time.time()

    # This loop does expensive float operations: ** 3.7 forces floating-point math
    for i in range(iterations):
        total += (task_id + i) ** 3.7

    # Optional: add a short sleep to extend visibility in dashboard
    # time.sleep(1)

    duration = time.time() - start_time
    print(f"Task {task_id} completed in {duration:.2f} seconds")

    return total


def run_ray_test():
    """Main test: submit multiple heavy tasks and wait for results."""
    initialize_ray()

    num_tasks = 200  # Submit more tasks than available CPUs to see queuing

    print(f"\nSubmitting {num_tasks} heavy pure-Python tasks...")
    print(
        "Check the Ray dashboard now — you should see multiple worker processes with high CPU usage!\n"
    )

    # Submit all tasks
    futures = [heavy_pure_python_task.remote(i) for i in range(num_tasks)]

    # Block until all complete and fetch results
    results = ray.get(futures)

    print(f"\nAll tasks completed. Sample results: {results[:5]}...")
    print("Ray test finished successfully.")


if __name__ == "__main__":
    try:
        run_ray_test()
    finally:
        # Always clean shutdown
        if ray.is_initialized():
            ray.shutdown()
        print("Ray shutdown complete.")
