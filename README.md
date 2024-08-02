# Llama 3.x One-Click Installation on VALDI

This project provides a simple, one-click installation script for Llama 3 models using the Hugging Face repository. It sets up necessary dependencies and downloads the specified Llama 3 model.

## Requirements

Before running the installation script, ensure you have the following:

- Python 3.8 or higher and pip
- A Hugging Face account with accepted Llama 3 license agreement
- Hugging Face access token with read permissions
- A Valdi GPU with Pytorch an CUDA installed

## Installation

1. Clone this repository or download the `install_llama3.py` script.
2. Open a terminal and navigate to the directory containing the script.
3. Run the installation script:

   ```
   python install_llama3.py --model "meta-llama/Meta-Llama-3.1-8B" --token <YOUR_HF_TOKEN>
   ```

   Replace `"meta-llama/Meta-Llama-3.1-8B"` with the actual model name and `YOUR_HF_TOKEN"` with your Hugging Face access token.
4. The script will install dependencies and download the model files. This process may take some time depending on your internet connection and the size of the model.

## Usage

After installation, you can use the Llama 3 model in your Python code as follows:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "meta-llama/Meta-Llama-3-8B"  # Use the appropriate model name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

def generate_text(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
  
    # Generate
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
  
    # Decode and return the result
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
prompt = "What color is the sky?"
result = generate_text(prompt)
print(result)


```

## Troubleshooting

- If you encounter permission issues when downloading the model, ensure you have the necessary access rights on Hugging Face for the specified model.
- If you face any issues with dependencies, try upgrading pip and setuptools:
  ```
  pip install --upgrade pip setuptools
  ```
- If the model files fail to download, double-check the model name and your access token.

## License

Please note that the Llama 3 models may be subject to specific license terms from Meta. Ensure you comply with these terms when using the model.

For any other issues or questions, please open an issue in this repository.
