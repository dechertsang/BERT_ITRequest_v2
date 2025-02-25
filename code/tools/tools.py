from transformers import AutoModel, AutoTokenizer

# Specify the model name or path
model_name = "bert-base-uncased"  # Or any other model identifier

# Download the model and tokenizer
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save the model and tokenizer locally (optional)
model.save_pretrained("./my_local_model")
tokenizer.save_pretrained("./my_local_model")