from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

torch.manual_seed(42)

text = "A new patient has had severe headaches and nausea. The headaches feel like throbbing pain on one side of his head, and they happen almost daily. The patient also feels very nauseous when the headaches come on, and sometimes the patient vomits from the nausea. The patient has been feeling excessively tired no matter how much rest they get. The patient sleeps over 8 hours every night, but still wakes up exhausted. The patient just moved to this area so you may not have the patient's previous medical records on file yet."

model_path = "xz97/AlpaCare-llama1-7b"
compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenize = AutoTokenizer.from_pretrained(model_path)
model_instance = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(compute_device)

input_prep = tokenize(text, return_tensors="pt", padding=True, truncation=True, max_length=1024)
input_to_model = input_prep.input_ids.to(compute_device)

with torch.no_grad():
    output_ids = model_instance.generate(input_to_model, max_length=256, num_return_sequences=1)
    outcome_text = tokenize.decode(output_ids[0], skip_special_tokens=True)

print(outcome_text)