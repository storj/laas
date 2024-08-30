import sys
import subprocess
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Llama 3 One-Click Installation Script")
    parser.add_argument("--model", type=str, required=True, help="Llama 3 model name")
    parser.add_argument("--token", type=str, required=True, help="Hugging Face access token")
    return parser.parse_args()

# this non-requirements.txt is simply for convienence
def check_and_install_libraries():
    required = ['transformers', 'torch', 'accelerate', 'huggingface_hub']
    
    for library in required:
        try:
            __import__(library)
            print(f"{library} is already installed.")
        except ImportError:
            print(f"Installing {library}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', library], 
                                  stdout=subprocess.DEVNULL, 
                                  stderr=subprocess.DEVNULL)
            print(f"{library} installed successfully.")

def download_model(model_name, token):
    from huggingface_hub import hf_hub_download, HfApi
    from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError

    api = HfApi()
    try:
        files = api.list_repo_files(repo_id=model_name, token=token)
        model_files = [f for f in files if f.endswith(('.json', '.model', '.safetensors', '.bin'))]
    except (RepositoryNotFoundError, RevisionNotFoundError):
        print(f"Error: Could not find repository or files for {model_name}")
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
    try:
        args = parse_arguments()
    except SystemExit:
        print("Error: Invalid command line arguments.")
        print("Usage: python script.py --model MODEL_NAME --token HF_TOKEN")
        return

    check_and_install_libraries()
    
    print(f"Attempting to download Llama 3 model: {args.model}")
    if download_model(args.model, args.token):
        print("Llama 3 installation complete!")
        print(f"You can now use the model by specifying '{args.model}' when loading with transformers")
    else:
        print("Installation failed. Please check the model name and your access token.")

if __name__ == "__main__":
    main()