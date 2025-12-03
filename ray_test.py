import ray, torch

ray.init()

@ray.remote(num_gpus=1)
def gpu_test():
    print("ray.get_gpu_ids():", ray.get_gpu_ids())
    x = torch.randn(1000, 1000, device="cuda")
    y = x @ x
    return y.sum().item()

result = ray.get(gpu_test.remote())
print("Result:", result)
