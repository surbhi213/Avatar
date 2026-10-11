import torch
STREAMING_VAE = True
# COMPILE = True
import os
COMPILE = os.getenv("ENABLE_COMPILE", "true").lower() == "true"
print(f"COMPILE: {COMPILE}")
torch._dynamo.config.cache_size_limit = 128

NO_REFRESH_INFERENCE = False

def is_compile_supported():
    return hasattr(torch, "compiler") and hasattr(torch.nn.Module, "compile")

def disable(func):
    if is_compile_supported():
        return torch.compiler.disable(func)
    return func

# COMPILE_DYNAMIC: "auto" (default) lets torch pick; "0" compiles one static graph per
# input shape, which avoids the slow shape-generic graph used while the KV window grows.
_DYNAMIC = {"0": False, "1": True}.get(os.getenv("COMPILE_DYNAMIC", "auto"), None)


def conditional_compile(func):
    if COMPILE:
        return torch.compile(mode=None, backend="inductor", dynamic=_DYNAMIC)(func)
    else:
        return func