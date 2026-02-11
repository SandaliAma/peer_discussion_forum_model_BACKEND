"""
Configuration file for Math RAG System with Fine-Tuning
D: Drive Version - Optimized for Intel i7-11800H with 16GB RAM
"""

import os
import torch
from dotenv import load_dotenv

load_dotenv()

# FORCE HUGGING FACE TO USE D: DRIVE (MUST BE FIRST!)

os.environ['HF_HOME'] = 'D:\\huggingface'
os.environ['TRANSFORMERS_CACHE'] = 'D:\\huggingface\\transformers'
os.environ['HF_DATASETS_CACHE'] = 'D:\\huggingface\\datasets'

# PATHS - D: DRIVE STRUCTURE

# BASE_DIR = "D:\\MathRAG"
BASE_DIR = r"D:\sliit\4.1\RESEARCH\RESEARCH_SYSTEM\model\MathRAG"

DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RAG_INDEX_DIR = os.path.join(BASE_DIR, "rag_index")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
TESTS_DIR = os.path.join(BASE_DIR, "tests")
VALIDATION_LOGS_DIR = os.path.join(LOGS_DIR, "validation_data")  # For training data collection

# Data file
# MATH_PROBLEMS_JSON = os.path.join(DATA_DIR, "math_problems.json")
MATH_PROBLEMS_JSON = os.path.join(DATA_DIR, "combined_dataset.json")
# Model paths
SINHALM_MODEL_PATH = os.path.join(MODELS_DIR, "SINHALM-SINHALA-GEMMA-3-4B-IT-FT")
# BASE_MODEL_NAME = "google/gemma-3-4b-it"
BASE_MODEL_NAME = "google/gemma-2-2b-it"
# Fine-tuned model path (will be created)
FINETUNED_MODEL_PATH = os.path.join(MODELS_DIR, "math_finetuned")

# CPU OPTIMIZATION - Use your 8 cores efficiently

torch.set_num_threads(6)  # Use 6 of 8 cores

# MODEL SETTINGS

# Embedding model (supports Sinhala)
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# Generation settings
MAX_LENGTH = 384  # Increased from 128 for complete Sinhala step-by-step explanations
TEMPERATURE = 0.7
TOP_P = 0.9
TOP_K = 30
REPETITION_PENALTY = 1.05

# RAG SETTINGS

NUM_EXAMPLES = 2  # Increased from 1 for better context
EMBEDDING_BATCH_SIZE = 1  # Increased for your good CPU

# FINE-TUNING SETTINGS

# Training parameters
FINETUNE_OUTPUT_DIR = FINETUNED_MODEL_PATH
FINETUNE_EPOCHS = 1
FINETUNE_BATCH_SIZE = 1
FINETUNE_GRADIENT_ACCUMULATION = 2
FINETUNE_LEARNING_RATE = 5e-4
FINETUNE_WARMUP_STEPS = 100
FINETUNE_SAVE_STEPS = 200
FINETUNE_LOGGING_STEPS = 10
FINETUNE_MAX_LENGTH = 512

# LoRA configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training data split
TRAIN_TEST_SPLIT = 0.9  # 90% train, 10% validation

# API SERVER SETTINGS

API_HOST = "0.0.0.0"
API_PORT = 5000
API_DEBUG = False
# timeout
GENERATION_TIMEOUT = 60

# GROQ API SETTINGS (for validation and fallback)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"  # Fast and accurate for math
GROQ_VALIDATION_ENABLED = True  # Set to False to disable validation
GROQ_TEMPERATURE = 0.3  # Lower for more consistent math answers
GROQ_DIRECT_MODE = False  # False = Use your fine-tuned model first, then Groq validates (SLOW but uses your training)

# MONGODB SETTINGS

MONGO_URL = os.getenv("MONGO_URL", "")
MONGO_DB_NAME = "math_forum"
MONGO_COLLECTION = "math_responses"

# LOGGING

LOG_LEVEL = "INFO"
LOG_FILE = os.path.join(LOGS_DIR, "math_rag.log")

# SYSTEM

DEVICE = "cpu"

# Create all directories
for directory in [DATA_DIR, MODELS_DIR, RAG_INDEX_DIR, LOGS_DIR, SCRIPTS_DIR, TESTS_DIR, VALIDATION_LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

os.makedirs(os.environ['HF_HOME'], exist_ok=True)

# VALIDATION

def validate_config():
    """Validate configuration"""
    import shutil
    
    errors = []
    warnings = []
    
    print("\n" + "="*70)
    print("CONFIGURATION VALIDATION")
    print("="*70)
    
    # Check Hugging Face cache
    print(f"\n📦 Hugging Face Cache: {os.environ['HF_HOME']}")
    
    # Check disk space
    total_d, used_d, free_d = shutil.disk_usage("D:\\")
    free_d_gb = free_d // (2**30)
    print(f"💾 D: Drive Free Space: {free_d_gb} GB")
    
    if free_d_gb < 15:
        errors.append(f"Not enough space on D: drive ({free_d_gb} GB free, need 15+ GB)")
    
    # Check data file
    if not os.path.exists(MATH_PROBLEMS_JSON):
        warnings.append(f"Data file not found: {MATH_PROBLEMS_JSON}")
    else:
        print(f"✓ Data file found: {MATH_PROBLEMS_JSON}")
    
    # Check SinhaLM model
    if not os.path.exists(SINHALM_MODEL_PATH):
        warnings.append(f"SinhaLM model not found: {SINHALM_MODEL_PATH}")
    else:
        print(f"✓ SinhaLM model found: {SINHALM_MODEL_PATH}")
    
    if errors:
        print("\n❌ ERRORS:")
        for e in errors:
            print(f"   - {e}")
        return False
    
    if warnings:
        print("\n⚠️  WARNINGS:")
        for w in warnings:
            print(f"   - {w}")
    
    print("\n" + "="*70)
    return True

if __name__ == "__main__":
    validate_config()