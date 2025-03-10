from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "/home/spshetty/RadAnnotate/finetune/v1/mistral-finetuned-v2/checkpoint-1084"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).eval().to("cpu")

vocab_size = tokenizer.vocab_size
seq_length = 512 

dummy_input = torch.randint(0, vocab_size, (1, seq_length)).to("cpu")

torch.onnx.export(
    model,
    dummy_input,
    "mistral_finetuned.onnx",
    opset_version=17,  
    input_names=["input"],
    output_names=["output"]
)

print("ONNX export completed with opset 17 on CPU")

