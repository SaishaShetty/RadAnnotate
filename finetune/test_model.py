import torch
from transformers import AutoTokenizer

# Paths to the fine-tuned model checkpoint and base model
checkpoint_path = "./mistral-finetuned/pytorch_model.bin"
base_model_path = "/home/spshetty/RadAnnotate/finetune/mixtral-7b"
fixed_checkpoint_path = "./mistral-finetuned-fixed/pytorch_model.bin"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer_vocab_size = len(tokenizer)

# Load the checkpoint
print("Loading checkpoint...")
state_dict = torch.load(checkpoint_path, map_location="cpu")

# Resize embed_tokens.weight and lm_head.weight
for key in ["model.embed_tokens.weight", "lm_head.weight"]:
    if key in state_dict:
        old_weight = state_dict[key]
        old_vocab_size, hidden_size = old_weight.size()

        # Resize weights to match tokenizer vocabulary size
        if tokenizer_vocab_size > old_vocab_size:
            print(f"Expanding {key} from {old_vocab_size} to {tokenizer_vocab_size}...")
            new_weight = torch.zeros((tokenizer_vocab_size, hidden_size))
            new_weight[:old_vocab_size, :] = old_weight
        elif tokenizer_vocab_size < old_vocab_size:
            print(f"Truncating {key} from {old_vocab_size} to {tokenizer_vocab_size}...")
            new_weight = old_weight[:tokenizer_vocab_size, :]
        else:
            print(f"No changes needed for {key}.")
            new_weight = old_weight

        state_dict[key] = new_weight

# Save the adjusted checkpoint
print("Saving fixed checkpoint...")
torch.save(state_dict, fixed_checkpoint_path)
print("Checkpoint saved at:", fixed_checkpoint_path)
