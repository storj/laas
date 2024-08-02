import argparse
import subprocess
from huggingface_hub import hf_hub_download, HfApi
from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError

def parse_arguments():
    parser = argparse.ArgumentParser(description="Llama 3 One-Click Installation Script")
    parser.add_argument("--model", type=str, required=True, help="Llama 3 model name")
    parser.add_argument("--token", type=str, required=True, help="Hugging Face access token")
    return parser.parse_args()

def install_dependencies():
    subprocess.check_call([
        "pip", "install", "--upgrade", "torch", "transformers", "accelerate", "huggingface_hub"
    ])

def get_model_files(model_name, token):
    api = HfApi()
    try:
        files = api.list_repo_files(repo_id=model_name, token=token)
        return [f for f in files if f.endswith(('.json', '.model', '.safetensors', '.bin'))]
    except (RepositoryNotFoundError, RevisionNotFoundError):
        print(f"Error: Could not find repository or files for {model_name}")
        return None

def download_model(model_name, token):
    model_files = get_model_files(model_name, token)
    if not model_files:
        return False

    for file in model_files:
        try:
            hf_hub_download(
                repo_id=model_name,
                filename=file,
                token=token,
                force_download=True
            )
            print(f"Downloaded: {file}")
        except Exception as e:
            print(f"Error downloading {file}: {str(e)}")
    
    return True

def main():
    args = parse_arguments()
    
    print("Installing dependencies...")
    install_dependencies()
    
    print(f"Attempting to download Llama 3 model: {args.model}")
    if download_model(args.model, args.token):
        print("Llama 3 installation complete!")
        print(f"You can now use the model by specifying '{args.model}' when loading with transformers")
    else:
        print("Installation failed. Please check the model name and your access token.")

if __name__ == "__main__":
    main()
