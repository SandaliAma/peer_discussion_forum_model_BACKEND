# SinhaLM Mathematics RAG System with Fine-Tuning

Complete Retrieval-Augmented Generation (RAG) system for Sinhala mathematics problems with fine-tuning support.

## 🎯 Features

- ✅ **RAG System**: Retrieves similar problems and generates step-by-step solutions
- ✅ **Fine-Tuning**: Train on your specific mathematics dataset
- ✅ **CPU Optimized**: Runs on Intel i7-11800H with 16GB RAM
- ✅ **Sinhala Support**: Full Unicode Sinhala language support
- ✅ **REST API**: Flask server for integration
- ✅ **D: Drive Storage**: Keeps C: drive clean

## 📋 System Requirements

- **CPU**: Intel i7-11800H (8 cores) ✅
- **RAM**: 16 GB ✅
- **Storage**: 15 GB free on D: drive ✅
- **OS**: Windows 10/11
- **Python**: 3.8 - 3.11

## 🚀 Quick Start

### 1. Run Installation Script

```batch
# Save INSTALL.bat to D:\MathRAG\
# Right-click → Run as Administrator
INSTALL.bat
```

### 2. Close and Reopen Terminal

⚠️ **IMPORTANT**: Close your terminal and open a new one for environment variables to take effect!

### 3. Copy Your Files

```batch
# Navigate to project
D:
cd D:\MathRAG

# Copy your data
copy "path\to\your\math_problems.json" data\

# Copy your SinhaLM model
xcopy "path\to\SINHALM-SINHALA-GEMMA-3-4B-IT-FT" models\SINHALM-SINHALA-GEMMA-3-4B-IT-FT\ /E /I
```

### 4. Verify Setup

```batch
python config.py
```

### 5. Run System

```batch
python run_system.py
```

## 📁 Project Structure

```
D:\MathRAG\
├── data/
│   └── math_problems.json              # Your mathematics problems
│
├── models/
│   ├── SINHALM-SINHALA-GEMMA-3-4B-IT-FT/   # Original SinhaLM
│   └── math_finetuned/                 # Fine-tuned model (created after training)
│
├── rag_index/
│   ├── faiss.index                     # Vector search index
│   └── problems.json                   # Processed problems
│
├── logs/
│   └── math_rag.log                    # System logs
│
├── venv/                               # Virtual environment
│
├── requirements.txt                    # Dependencies
├── config.py                           # Configuration
├── math_rag_system.py                  # Core RAG system
├── fine_tuner.py                       # Fine-tuning system
├── quick_test.py                       # Testing script
├── api_server.py                       # REST API server
└── run_system.py                       # Main runner

D:\huggingface\                         # Hugging Face models cache
└── transformers/
    └── models--google--gemma-3-4b-it/  # Base model (~9 GB)
```

## 🎮 Usage Guide

### Option 1: Interactive Menu (Recommended)

```batch
python run_system.py
```

**Menu Options:**
1. **Quick Test** - Test the system interactively
2. **Fine-Tune Model** - Train on your data
3. **Start API Server** - Run REST API
4. **Rebuild RAG Index** - Rebuild search index
5. **System Info** - View system status
6. **Compare Models** - Compare original vs fine-tuned
7. **Exit**

### Option 2: Direct Commands

#### Test the System

```batch
python quick_test.py
```

#### Fine-Tune the Model

```batch
python fine_tuner.py
```

**Expected Time:** 1-4 hours on CPU (depends on data size)

#### Start API Server

```batch
python api_server.py
```

Access at: `http://localhost:5000`

## 🎓 Fine-Tuning Process

### Step-by-Step Fine-Tuning

```batch
# 1. Activate environment
D:
cd D:\MathRAG
venv\Scripts\activate

# 2. Run fine-tuning
python fine_tuner.py

# OR use menu
python run_system.py
# Select option 2: Fine-Tune Model
```

### What Happens During Fine-Tuning

```
[STEP 1/5] Loading mathematics problems...
✓ Loaded 150 problems

[STEP 2/5] Preparing training dataset...
✓ Dataset prepared
   Train: 135 examples
   Validation: 15 examples

[STEP 3/5] Initializing fine-tuner...

[STEP 4/5] Preparing model...
   Loading base model: google/gemma-3-4b-it
   Loading SinhaLM adapter
   Adding new LoRA layers for mathematics
   Trainable parameters: 8,388,608
   Total parameters: 4,000,000,000
   Trainable %: 0.21%

[STEP 5/5] Fine-tuning...
   Training steps: 50
   Estimated time: 2-3 hours on CPU

Training: [████████████████████] 100%
Epoch 1/3 | Loss: 1.234
Epoch 2/3 | Loss: 0.987
Epoch 3/3 | Loss: 0.765

✅ TRAINING COMPLETE!
Model saved to: D:\MathRAG\models\math_finetuned
```

### Fine-Tuning Configuration

Edit `config.py` to adjust:

```python
# Fine-tuning parameters
FINETUNE_EPOCHS = 3              # Number of training epochs
FINETUNE_BATCH_SIZE = 1          # Batch size (increase if you have more RAM)
FINETUNE_LEARNING_RATE = 5e-4    # Learning rate
FINETUNE_GRADIENT_ACCUMULATION = 8  # Effective batch size multiplier

# LoRA parameters
LORA_R = 16                      # LoRA rank (higher = more parameters)
LORA_ALPHA = 32                  # LoRA alpha
LORA_DROPOUT = 0.1               # Dropout rate
```

## 🌐 API Documentation

### Endpoints

#### 1. Health Check

```bash
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "num_problems": 150,
  "model": "SinhaLM-Gemma-3-4b-it"
}
```

#### 2. Answer Question

```bash
POST /api/answer
Content-Type: application/json

{
  "question": "x² + 5x + 6 = 0 විසඳන්න",
  "num_examples": 3,
  "include_steps": true
}
```

**Response:**
```json
{
  "status": "success",
  "answer": "විසඳුම:\n(x+2)(x+3) = 0\nx = -2 හෝ x = -3",
  "similar_problems": [...],
  "num_retrieved": 3,
  "model_used": "fine-tuned",
  "timestamp": "2025-01-15T12:00:00"
}
```

#### 3. Search Similar Problems

```bash
POST /api/search
Content-Type: application/json

{
  "query": "සමීකරණ",
  "k": 5
}
```

#### 4. Get Topics

```bash
GET /api/topics
```

#### 5. Get Statistics

```bash
GET /api/stats
```

### API Usage Examples

**Python:**
```python
import requests

response = requests.post('http://localhost:5000/api/answer', json={
    'question': 'x² + 5x + 6 = 0 විසඳන්න',
    'num_examples': 3
})

result = response.json()
print(result['answer'])
```

**cURL:**
```bash
curl -X POST http://localhost:5000/api/answer \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"x² + 5x + 6 = 0 විසඳන්න\"}"
```

## 📊 Expected Performance

| Operation | First Time | Subsequent |
|-----------|-----------|------------|
| Model load | 2-3 min | 30-60 sec |
| Index build | 10-30 sec | Instant (cached) |
| Per question | 3-8 sec | 3-8 sec |
| Fine-tuning | 1-4 hours | - |

## 🔧 Configuration

### Key Settings in `config.py`

```python
# Paths
BASE_DIR = "D:\\MathRAG"
MATH_PROBLEMS_JSON = "D:\\MathRAG\\data\\math_problems.json"

# Model settings
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
MAX_LENGTH = 1024
TEMPERATURE = 0.7

# RAG settings
NUM_EXAMPLES = 3
EMBEDDING_BATCH_SIZE = 32

# CPU optimization
torch.set_num_threads(6)  # Use 6 of your 8 cores
```

## 🐛 Troubleshooting

### Issue: Model Downloads to C: Drive

**Solution:**
```batch
# Check environment variables
echo %HF_HOME%
# Should show: D:\huggingface

# If not, close terminal and reopen
# Or set manually:
set HF_HOME=D:\huggingface
set TRANSFORMERS_CACHE=D:\huggingface\transformers
```

### Issue: Out of Memory

**Solution:**
```python
# In config.py, reduce:
EMBEDDING_BATCH_SIZE = 16  # Reduce from 32
MAX_LENGTH = 512  # Reduce from 1024
FINETUNE_BATCH_SIZE = 1  # Keep at 1
```

### Issue: Slow Generation

**Solution 1 - Reduce max length:**
```python
MAX_LENGTH = 512  # In config.py
```

**Solution 2 - Use greedy decoding:**
```python
# In math_rag_system.py, _generate method:
do_sample=False,  # Instead of True
temperature=0,  # Instead of 0.7
```

### Issue: Fine-Tuning Too Slow

**Solution - Use Google Colab (Free GPU):**
1. Upload your files to Google Drive
2. Use Colab with GPU (free)
3. Fine-tuning takes 15-30 minutes instead of 2-4 hours
4. Download fine-tuned model back to D:\MathRAG\models\

### Issue: Sinhala Text Not Displaying

**Solution:**
```python
# Add at top of scripts
import sys
sys.stdout.reconfigure(encoding='utf-8')
```

Use Windows Terminal or VSCode terminal for proper Unicode support.

## 📈 Improving Results

### 1. Add More Training Data

- Minimum: 50 problems
- Recommended: 200+ problems
- Best: 500+ problems

### 2. Fine-Tune the Model

- Original SinhaLM: Good for general Sinhala
- Fine-tuned: Better for your specific math problems

### 3. Adjust RAG Settings

```python
NUM_EXAMPLES = 5  # Increase from 3 for more context
```

### 4. Improve Search Quality

- Add more diverse examples
- Include varied problem types
- Cover different difficulty levels

## 📚 Data Format

Your `math_problems.json` should follow this structure:

```json
{
  "examples": [
    {
      "type": "exercises",
      "topic": "සමීකරණය",
      "sub_topic": "වර්ගජ සමීකරණ",
      "question": "x² + 5x + 6 = 0 විසඳන්න",
      "steps": [
        {
          "step_description": "සාධක භාවිතයෙන්",
          "step_answer": "(x+2)(x+3) = 0"
        },
        {
          "step_description": "x + 2 = 0 හෝ x + 3 = 0",
          "step_answer": "x = -2 හෝ x = -3"
        }
      ],
      "final_answer": "x = -2 හා x = -3"
    }
  ]
}
```

## 🔍 Validation

### Check Configuration
```batch
python config.py
```

### Check Data
```batch
python verify_data.py
```

### Test System
```batch
python quick_test.py
```

## 💾 Disk Space Usage

| Component | Size |
|-----------|------|
| Python packages | ~2-3 GB |
| Base Gemma model | ~9 GB |
| SinhaLM adapter | ~200 MB |
| Embedding model | ~420 MB |
| Fine-tuned adapter | ~200 MB |
| RAG index | ~100 MB |
| **Total** | **~12-13 GB** |

## 🚦 Status Indicators

- 🟢 **Ready**: System is operational
- 🟡 **Building**: Loading models or building index
- 🔵 **Training**: Fine-tuning in progress
- 🟠 **Warning**: Low memory or disk space
- 🔴 **Error**: Check logs for details

## 📞 Support

For issues:
1. Check logs: `D:\MathRAG\logs\math_rag.log`
2. Verify configuration: `python config.py`
3. Check disk space: Look for warnings
4. Restart system: Close and reopen terminal

## 📄 License

This project uses:
- SinhaLM: Apache 2.0
- Gemma: Google's Terms
- Sentence Transformers: Apache 2.0

## 🎉 Success Indicators

You're ready when you see:

```
✅ SYSTEM BUILD COMPLETE!
✅ Configuration is valid!
✅ Index built successfully!
✅ Model loaded successfully!
✅ System ready!
```

## 🚀 Next Steps After Installation

1. **Test with sample questions** - Verify system works
2. **Add your problems** - Build your dataset
3. **Fine-tune (optional)** - Improve accuracy
4. **Deploy API** - Integrate with your application

---

**Version**: 1.0.0  
**Last Updated**: January 2025  
**Optimized for**: Intel i7-11800H, 16GB RAM, Windows 11





Database: math_forum | Collection: math_responses


{
  "student_id": "test_user",
  "question": "x + 5 = 10 නම් x හි අගය කීයද?",
  "answer": "step by step solution...",
  "source": "groq_improved",
  "model_used": "fine-tuned",
  "response_time_ms": 2500,
  "validation": { "is_correct": true, "confidence": 0.95 },
  "similar_problems_count": 2,
  "created_at": "2026-02-08T..."
}

config.py	Configuration
fast_api_server.py	Main API server
groq_validator.py	Groq validation (current)
math_rag_system.py	Core RAG system
fine_tuner.py	For future fine-tuning
requirements.txt	Dependencies
test_groq.py	Test Groq API