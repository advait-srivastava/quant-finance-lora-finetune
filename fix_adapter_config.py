"""Make an Unsloth-MLX adapter loadable by plain mlx_lm.

Unsloth's save_lora_adapters() writes its own config schema (quantization maps,
unsloth_mlx_lora_module_paths). mlx_lm.load() instead needs fine_tune_type /
num_layers / lora_parameters, so it fails with:
    AttributeError: 'types.SimpleNamespace' object has no attribute 'num_layers'

The weights themselves are already fine (lora_a/lora_b names match mlx_lm's
LoRALinear), so this only adds the missing fields. Unsloth's own keys are kept
so its loader still works.

scale MUST be lora_alpha/r (1.0 here), not mlx_lm's 20.0 default, or the
adapter is applied 20x too strong.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

from safetensors import safe_open

ADAPTER = Path(sys.argv[1] if len(sys.argv) > 1 else "./fo-adapter-unsloth")
R = 16
LORA_ALPHA = 16

cfg_path = ADAPTER / "adapter_config.json"
cfg = json.loads(cfg_path.read_text())

# Derive layer count and module keys from the weights, so this stays correct
# if the training config changes.
with safe_open(str(ADAPTER / "adapters.safetensors"), "numpy") as f:
    tensor_names = list(f.keys())

layer_ids = {int(m.group(1)) for n in tensor_names
             if (m := re.search(r"layers\.(\d+)\.", n))}
num_layers = max(layer_ids) + 1

keys = sorted({
    m.group(1) for n in tensor_names
    if (m := re.search(r"layers\.\d+\.(.+)\.lora_[ab]$", n))
})

cfg.update({
    "fine_tune_type": "lora",
    "num_layers": num_layers,
    "lora_parameters": {
        "rank": R,
        "scale": LORA_ALPHA / R,
        "dropout": 0.0,
        "keys": keys,
    },
})

cfg_path.write_text(json.dumps(cfg, indent=2))

print(f"Patched {cfg_path}")
print(f"  num_layers : {num_layers} (layers {min(layer_ids)}-{max(layer_ids)})")
print(f"  scale      : {LORA_ALPHA / R}")
print(f"  keys       : {len(keys)} modules")
for k in keys:
    print(f"     {k}")
print(f"  tensors    : {len(tensor_names)}")
print(f"  patterns   : {Counter(n.split('.')[-1] for n in tensor_names)}")
