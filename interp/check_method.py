
from transformer_lens import HookedTransformer
print("Checking HookedTransformer methods...")
print(f"Has set_use_attn_result: {hasattr(HookedTransformer, 'set_use_attn_result')}")
