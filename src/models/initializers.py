import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from sentence_transformers import SentenceTransformer
from ..config import QWEN_DEVICE, VIDEOLLAMA_DEVICE, EMBEDDING_DEVICE


def initialize_qwen_model(device=None):
    if device is None:
        device = QWEN_DEVICE
    print(f"Initializing Qwen model on {device}...")
    model_name = "Qwen/Qwen3-32B-AWQ"
    
    qwen_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map={"": device}
    )
    qwen_tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Qwen model initialized")
    return qwen_model, qwen_tokenizer


def initialize_embedding_model(device=None):
    if device is None:
        device = EMBEDDING_DEVICE
    print(f"Initializing embedding model on {device}...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    print("Embedding model initialized")
    return embedding_model


def initialize_videollama3(device=None):
    if device is None:
        device = VIDEOLLAMA_DEVICE
    print(f"Initializing VideoLLaMA3 model on {device}...")
    model_path = "DAMO-NLP-SG/VideoLLaMA3-7B"
    
    videollama_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map={"": device},
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    videollama_processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True
    )
    print("VideoLLaMA3 initialized")
    return videollama_model, videollama_processor
