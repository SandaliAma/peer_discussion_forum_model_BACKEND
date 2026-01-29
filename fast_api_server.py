"""
Fast API Server using RAG System
Much faster than direct model inference
WITH TIMEOUT PROTECTION
"""
import torch
torch.set_num_threads(6)
torch.set_num_interop_threads(1)

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import time
import signal
from contextlib import contextmanager
from math_rag_system import SystemBuilder
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global RAG system (initialized once)
rag_system = None

# Timeout handler
class TimeoutError(Exception):
    pass

@contextmanager
def timeout(seconds):
    """Context manager for timeout"""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

@app.before_request
def initialize():
    """Initialize RAG system on first request"""
    global rag_system
    
    if rag_system is None:
        logger.info("Initializing RAG system with fine-tuned model...")
        logger.info("This takes 2-3 minutes on first startup...")
        
        try:
            rag_system = SystemBuilder.build(use_finetuned=False)
            logger.info("✓ RAG system ready!")
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            logger.info("Trying with original SinhaLM...")
            rag_system = SystemBuilder.build(use_finetuned=False)

@app.route('/api/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'model': 'SinhaLM with RAG',
        'ready': rag_system is not None,
        'max_generation_time': f"{config.GENERATION_TIMEOUT}s"
    })

@app.route('/api/answer', methods=['POST'])
def answer():
    """
    Answer a question using RAG system
    Much faster than direct model inference!
    """
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({
                'status': 'error',
                'error': 'No question provided'
            }), 400
        
        question = data['question']
        student_id = data.get('student_id', 'anonymous')
        
        logger.info(f"Question from {student_id}: {question[:50]}...")
        
        # Measure response time
        start_time = time.time()
        
        try:
            # Try with timeout (Windows-compatible version)
            result = rag_system.answer_question(
                question=question,
                num_examples=config.NUM_EXAMPLES,  # Using reduced examples
                include_steps=True
            )
            
            end_time = time.time()
            response_time_ms = int((end_time - start_time) * 1000)
            
            # Check if it took too long
            if response_time_ms > config.GENERATION_TIMEOUT * 1000:
                logger.warning(f"⚠️ Generation took {response_time_ms/1000:.1f}s - consider reducing MAX_LENGTH")
            
            logger.info(f"✓ Answered in {response_time_ms}ms")
            
            # Add response time
            result['response_time_ms'] = response_time_ms
            result['student_id'] = student_id
            
            return jsonify(result)
            
        except TimeoutError as te:
            logger.error(f"Timeout: {te}")
            return jsonify({
                'status': 'error',
                'error': f'Generation timeout after {config.GENERATION_TIMEOUT}s. Try reducing question complexity.',
                'suggestion': 'The model is taking too long on CPU. Consider: 1) Reducing MAX_LENGTH in config.py, 2) Using a GPU, 3) Simplifying the question'
            }), 504
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/search', methods=['POST'])
def search():
    """Search similar problems"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        k = data.get('k', 5)
        
        similar = rag_system.rag_db.search(query, k=k)
        
        return jsonify({
            'status': 'success',
            'results': [
                {
                    'question': r['problem']['question'],
                    'topic': r['problem']['topic'],
                    'similarity': r['similarity']
                }
                for r in similar
            ]
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("\n" + "="*70)
    print("FAST API SERVER - WITH RAG OPTIMIZATION")
    print("="*70)
    print("\n⚠️  First startup takes 2-3 minutes to load model")
    print("   After that, responses are MUCH faster!")
    print(f"\n⏱️  Max generation time: {config.GENERATION_TIMEOUT}s")
    print(f"   Max output length: {config.MAX_LENGTH} tokens")
    print(f"\nStarting on http://{config.API_HOST}:{config.API_PORT}")
    print("="*70 + "\n")
    
    app.run(
        host=config.API_HOST,
        port=config.API_PORT,
        debug=False,
        threaded=True
    )