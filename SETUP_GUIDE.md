# MathRAG with Gemini Validation - Setup Guide

##  What's New?

Your system now uses a **hybrid approach**:
1. **Fine-tuned model** generates initial answer
2. **Gemini API** validates and improves the answer
3. **Best answer** is returned to students with step-by-step solutions

This ensures **correct answers** while you continue improving your fine-tuned model.

---

##  Prerequisites

1. Python 3.8+
2. Gemini API key from Google AI Studio

---

##  Setup Instructions

### Step 1: Install Dependencies

```bash
pip install google-generativeai
```

Or install all dependencies:

```bash
pip install -r requirements.txt
```

### Step 2: Get Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Click **"Get API Key"**
3. Create a new API key
4. Copy the key

### Step 3: Configure Gemini API

Open `config.py` and add your API key:

```python
# GEMINI API SETTINGS (for validation and fallback)

GEMINI_API_KEY = "YOUR_API_KEY_HERE"  # Add your key here
GEMINI_MODEL = "gemini-1.5-flash"  # Fast and cost-effective
GEMINI_VALIDATION_ENABLED = True  # Set to False to disable
GEMINI_TEMPERATURE = 0.3  # Lower for consistent math answers
```

**Important:** Replace `"YOUR_API_KEY_HERE"` with your actual API key!

### Step 4: Start the Server

```bash
python fast_api_server.py
```

You should see:

```
======================================================================
MATH RAG API SERVER - FINE-TUNED + GEMINI VALIDATION
======================================================================

  First startup takes 2-3 minutes to load model
   After that, responses are MUCH faster!

 Configuration:
   Max generation time: 60s
   Max output length: 384 tokens
   RAG examples: 2
   Gemini validation: ✓ ENABLED



 Flow: Fine-tuned Model → Gemini Validation → Best Answer
======================================================================
```

---

## Testing

### Test 1: Health Check

**Endpoint:** `GET http://localhost:5000/api/health`

**Expected Response:**
```json
{
  "status": "healthy",
  "model": "SinhaLM Fine-tuned with RAG + Gemini Validation",
  "ready": true,
  "validation_enabled": true,
  "max_generation_time": "60s",
  "max_length": 384,
  "num_examples": 2
}
```

### Test 2: Ask a Math Question

**Endpoint:** `POST http://localhost:5000/api/answer`

**Request Body:**
```json
{
  "question": "x + 5 = 10 නම් x හි අගය කීයද?",
  "student_id": "test_student"
}
```

**Expected Response:**
```json
{
  "status": "success",
  "answer": "පියවරෙන් පියවර විසඳුම:\n\n1. සමීකරණය: x + 5 = 10\n2. දෙපසින් 5 අඩු කරන්න:\n   x + 5 - 5 = 10 - 5\n3. සරල කරන්න:\n   x = 5\n\nඅවසාන පිළිතුර: x = 5",
  "source": "gemini_improved",
  "response_time_ms": 2500,
  "student_id": "test_student",
  "model_used": "fine-tuned",
  "validation": {
    "enabled": true,
    "is_correct": true,
    "confidence": 0.95,
    "final_answer": "x = 5"
  },
  "similar_problems_count": 2
}
```

### Test 3: Postman Collection

Import this JSON into Postman:

```json
{
  "info": {
    "name": "MathRAG API",
    "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
  },
  "item": [
    {
      "name": "Health Check",
      "request": {
        "method": "GET",
        "header": [],
        "url": "http://localhost:5000/api/health"
      }
    },
    {
      "name": "Answer Question",
      "request": {
        "method": "POST",
        "header": [
          {
            "key": "Content-Type",
            "value": "application/json"
          }
        ],
        "body": {
          "mode": "raw",
          "raw": "{\n  \"question\": \"2x + 4 = 12 නම් x කීයද?\",\n  \"student_id\": \"student_001\"\n}"
        },
        "url": "http://localhost:5000/api/answer"
      }
    }
  ]
}
```

---

## 📊 How It Works

### The Validation Flow

```
Student asks question
        ↓
Fine-tuned Model generates answer (2-5 seconds)
        ↓
Gemini validates and improves (1-3 seconds)
        ↓
Return best answer with step-by-step solution
        ↓
Log for future training data
```

### Response Fields Explained

- **answer**: The final step-by-step solution in Sinhala
- **source**: Where the answer came from
  - `gemini_improved`: Gemini validated/improved the model's answer
  - `finetuned_model`: Used model answer (when validation disabled)
  - `finetuned_model_fallback`: Gemini failed, using model answer
- **validation.is_correct**: Whether the model's original answer was correct
- **validation.confidence**: Gemini's confidence in the answer (0-1)
- **validation.final_answer**: The numerical final answer
- **similar_problems_count**: How many similar problems were retrieved from RAG

---

## 💰 Cost Management

### Gemini API Pricing (as of 2024)

- **Gemini 1.5 Flash**: Very cheap, ~$0.000075 per request
- **Free tier**: 15 requests per minute, 1500 per day

### Cost Estimation

For 1000 student questions per day:
- With validation: ~$0.08 per day (~$2.40/month)
- Very affordable for research/small scale

### Tips to Reduce Costs

1. **Disable validation** when not needed:
   ```python
   GEMINI_VALIDATION_ENABLED = False
   ```

2. **Use cached responses** for common questions

3. **Monitor usage** in Google AI Studio

---

## 📈 Collecting Training Data

All validation results are automatically logged to:
```
logs/validation_data/YYYYMMDD_validations.jsonl
```

### Using Logs for Retraining

1. Review logged data:
```bash
cat logs/validation_data/*_validations.jsonl | jq .
```

2. Filter incorrect predictions:
```bash
cat logs/validation_data/*_validations.jsonl | jq 'select(.used_gemini == true)'
```

3. Use this data to:
   - Identify weak areas in your model
   - Create better training examples
   - Fine-tune again with improved data

---

## 🔧 Configuration Options

### In `config.py`

```python
# Disable validation entirely
GEMINI_VALIDATION_ENABLED = False

# Use different Gemini model
GEMINI_MODEL = "gemini-1.5-pro"  # More accurate but slower/costlier

# Adjust response length
MAX_LENGTH = 512  # For longer explanations

# More RAG examples
NUM_EXAMPLES = 3  # Retrieve 3 similar problems
```

---

## ❓ Troubleshooting

### Issue: "Gemini API key not provided"

**Fix:** Add your API key to `config.py`:
```python
GEMINI_API_KEY = "your-actual-api-key-here"
```

### Issue: "Gemini validation failed"

**Possible causes:**
1. No internet connection
2. Invalid API key
3. Rate limit exceeded (wait a minute)

**Fix:** The system will automatically fall back to model answer

### Issue: "Fine-tuned model not found"

**Fix:** Run fine-tuning first:
```bash
python fine_tuner.py
```

Or temporarily use the base model by editing line 58 in `fast_api_server.py`:
```python
rag_system = SystemBuilder.build(use_finetuned=False)
```

### Issue: Slow responses

**Causes:**
1. Model generating on CPU (expected 2-5s)
2. Gemini API network delay (1-3s)

**Total expected time:** 3-8 seconds per question

This is normal for CPU-based systems with validation!

---

## 🎓 Next Steps

### Option 1: Keep Using Validation (Recommended for now)

Continue using the hybrid system while:
1. Collecting validation logs
2. Identifying common failure patterns
3. Building better training dataset

### Option 2: Improve Fine-tuned Model

After collecting enough data:
1. Review `logs/validation_data/`
2. Use Gemini to generate more training examples
3. Re-fine-tune with 500-1000 high-quality examples
4. Gradually reduce dependency on Gemini

### Option 3: Generate Training Data with Gemini

Use Gemini to create a large, high-quality dataset:
```bash
python generate_training_data.py  # You'll need to create this script
```

---

## 📞 Support

If you encounter issues:
1. Check the logs in `logs/math_rag.log`
2. Verify Gemini API key is correct
3. Test Gemini separately:
   ```bash
   python gemini_validator.py
   ```

---

## ✅ Summary

You now have:
- ✅ Fine-tuned model enabled
- ✅ Gemini validation system
- ✅ Step-by-step solutions in Sinhala
- ✅ Automatic logging for improvement
- ✅ Error handling and fallbacks

**Your system will provide correct answers while you continue improving your model!**
