import torch
from torch.profiler import profile, record_function, ProfilerActivity

import pyspiel
from icecream import ic


dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

a = torch.tensor([1], device=dev)
b = torch.tensor([2], device=dev)
c = torch.tensor([3], device=dev)
d = 4
e = torch.tensor([5], device=dev)
f = torch.tensor([6], device=torch.device("cpu"))

cumz = torch.tensor([a, b, c, d, e], device=dev)
some_tensor = torch.tensor([1, 1, 1, 1, 1], device=dev)

# Will this work?
tot = cumz + some_tensor

print(tot)
ic(type(tot))
ic(tot.device)

some_other_test = c + d
print(some_other_test)


import torch
from torch.profiler import profile, record_function, ProfilerActivity

# Create input data
x = torch.randn(10, device="cuda")
tensor = torch.tensor([0.5], device="cuda")

def warmup():
    for _ in range(10):
        _ = x * tensor
    torch.cuda.synchronize()

# Define a simple float conversion function
def float_conversion(x):
    y = x * 0.5
    return y

def tensor_multiply(x):
    y = x * tensor
    return y

warmup()

# Profiling for float conversion
with profile(
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
) as prof1:
    for _ in range(5):
        with record_function("float_conversion"):
            y = float_conversion(x)
        prof1.step()
print(prof1.key_averages().table(sort_by="cuda_time_total"))

# Profiling for tensor multiplication
with profile() as prof2:
    with record_function("tensor_multiply"):
        y = tensor_multiply(x)
print(prof2.key_averages().table(sort_by="cuda_time_total"))


with profile() as prof3:
    with record_function("float_conversion_v2"):
        y = float_conversion(x)
print(prof3.key_averages().table(sort_by="cuda_time_total"))


floaty = 0.5
floaty_tensor = torch.tensor([0.5], device=dev)
std_tensor = torch.tensor([6], device=dev)
std_float = 6

def add_tensors_conversion():
    for _ in range(1000):
        _ = floaty + std_tensor
    torch.cuda.synchronize()

def add_tensors_tensor():
    for _ in range(1000):
        _ = floaty_tensor + std_tensor
    torch.cuda.synchronize()

def python_addition():
    for _ in range(1000):
        _ = floaty + std_float
    torch.cuda.synchronize()

with profile() as prof4:
    with record_function("add_tensors_conversion"):
        add_tensors_conversion()
print(prof4.key_averages().table(sort_by="cuda_time_total"))

with profile() as prof5:
    with record_function("add_tensors_tensor"):
        add_tensors_tensor()
print(prof5.key_averages().table(sort_by="cuda_time_total"))

with profile() as prof6:
    with record_function("python_addition"):
        python_addition()
print(prof6.key_averages().table(sort_by="cuda_time_total"))
