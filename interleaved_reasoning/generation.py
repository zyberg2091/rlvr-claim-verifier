"""Reusable response generation from the interleaved reasoning notebook."""

import torch

# Use the same device selection as the original notebook.
device = "cuda" if torch.cuda.is_available() else "cpu"


def generate_response(prompt_ids, model, tokenizer, max_new_tokens=256):
    model.eval()

    generated_ids = []
    past_key_values = None
    current_ids = prompt_ids.to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, value, past_key_values = model(current_ids, past_key_values=past_key_values, use_cache=True)

            # Greedy decoding
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

            if next_token_id.item() == tokenizer.eos_token_id:
                break

            generated_ids.append(next_token_id.item())
            current_ids = next_token_id

    return tokenizer.decode(generated_ids, skip_special_tokens=True)
