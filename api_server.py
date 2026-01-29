"""
API Server for SinhaLM Math RAG Fine-Tuned Model
"""

import os
import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import config

# ============================================================================ 
# Flask App
# ============================================================================ 

app = Flask(__name__)

# ============================================================================ 
# Load Fine-Tuned Model
# ============================================================================ 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading fine-tuned model on {DEVICE}...")

tokenizer = AutoTokenizer.from_pretrained(config.FINETUNE_OUTPUT_DIR)
base_model = AutoModelForCausalLM.from_pretrained(
    config.BASE_MODEL_NAME,
    device_map="auto" if DEVICE.type == "cuda" else None,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)

if os.path.exists(config.FINETUNE_OUTPUT_DIR):
    model = PeftModel.from_pretrained(base_model, config.FINETUNE_OUTPUT_DIR)
    print("✓ Fine-tuned MathRAG model loaded")
else:
    print("⚠️ Fine-tuned model not found, using base model")
    model = base_model

model.to(DEVICE)
model.eval()

# ============================================================================ 
# Utility Function to Generate Answers
# ============================================================================ 

def generate_answer(question: str, max_length=1024, temperature=0.7, top_p=0.9, top_k=50):
    """
    Generate step-by-step answer using fine-tuned model
    """
    input_text = f"ප්‍රශ්නය: {question}\n\nමෙම ගණිත ගැටලුව පියවරෙන් පියවර විසඳන්න.\n\nවිසඳුම:"
    
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    ).to(DEVICE)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=config.REPETITION_PENALTY,
            do_sample=True
        )
    
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Remove input prompt to get only model's output
    if answer.startswith(input_text):
        answer = answer[len(input_text):].strip()
    
    return answer

# ============================================================================ 
# API Endpoints
# ============================================================================ 

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "database": "connected",
        "model": "fine-tuned MathRAG"
    })

@app.route("/api/answer", methods=["POST"])
def answer():
    data = request.get_json()
    
    if not data or "question" not in data:
        return jsonify({"status": "error", "message": "No question provided"}), 400
    
    question = data["question"]
    student_id = data.get("student_id", "anonymous")
    
    # Generate answer using fine-tuned model
    try:
        answer_text = generate_answer(question)
        return jsonify({
            "status": "success",
            "qa_id": 0,  # You can integrate DB ID here if needed
            "answer": answer_text,
            "model_used": "fine-tuned",
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# ============================================================================ 
# Main
# ============================================================================ 

if __name__ == "__main__":
    print("\n======================================================================")
    print("SINHALM MATH RAG API SERVER WITH FINE-TUNED MODEL")
    print("======================================================================")
    print(f"Starting server on http://{config.API_HOST}:{config.API_PORT}")
    
    app.run(host=config.API_HOST, port=config.API_PORT, debug=config.API_DEBUG)
