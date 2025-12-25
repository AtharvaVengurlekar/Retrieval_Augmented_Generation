from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

MODEL_NAME = "Your Llama-Based Model Name Here"
MODEL_DIR = "Your/Local/Directory/To/Save/The/Model"
os.makedirs(MODEL_DIR, exist_ok=True)

HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("ERROR: Please set your HuggingFace token as HF_TOKEN environment variable.")

print(f"Downloading {MODEL_NAME} ...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    cache_dir=MODEL_DIR,
    use_auth_token=HF_TOKEN
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    cache_dir=MODEL_DIR,
    use_auth_token=HF_TOKEN
)

print(f"Model downloaded successfully at '{MODEL_DIR}'")




from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# Model name
MODEL_NAME = "Your Large Language Model Name Here"

# Local directory to save the model
MODEL_DIR = "Your/Local/Directory/To/Save/The/Model"
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"Downloading {MODEL_NAME} ...")

# Download tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    cache_dir=MODEL_DIR
)

# Download model (FP16 by default if supported)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    cache_dir=MODEL_DIR
)

print(f"{MODEL_NAME} downloaded successfully at '{MODEL_DIR}'")
