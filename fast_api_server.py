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
from gemini_validator import GeminiValidator, ValidationLogger
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global RAG system and validator (initialized once)
rag_system = None
gemini_validator = None

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
    """Initialize RAG system and Gemini validator on first request"""
    global rag_system, gemini_validator

    if rag_system is None:
        logger.info("Initializing RAG system with fine-tuned model...")
        logger.info("This takes 2-3 minutes on first startup...")

        try:
            rag_system = SystemBuilder.build(use_finetuned=True)
            logger.info("✓ RAG system ready with fine-tuned model!")
        except Exception as e:
            logger.error(f"Failed to initialize fine-tuned model: {e}")
            logger.info("Falling back to original SinhaLM...")
            rag_system = SystemBuilder.build(use_finetuned=False)

        # Initialize Gemini validator if enabled
        if config.GEMINI_VALIDATION_ENABLED and config.GEMINI_API_KEY:
            try:
                gemini_validator = GeminiValidator()
                logger.info("✓ Gemini validator initialized!")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini validator: {e}")
                logger.warning("Continuing without validation")
                gemini_validator = None
        else:
            logger.info("Gemini validation disabled")
            gemini_validator = None

@app.route('/api/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'model': 'SinhaLM Fine-tuned with RAG + Gemini Validation',
        'ready': rag_system is not None,
        'validation_enabled': gemini_validator is not None,
        'max_generation_time': f"{config.GENERATION_TIMEOUT}s",
        'max_length': config.MAX_LENGTH,
        'num_examples': config.NUM_EXAMPLES
    })

@app.route('/api/answer', methods=['POST'])
def answer():
    """
    Answer a question using RAG system + Gemini validation
    Flow: Model generates -> Gemini validates & improves -> Return best answer
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
            # STEP 1: Get answer from fine-tuned model
            logger.info("Step 1: Getting answer from fine-tuned model...")
            model_result = rag_system.answer_question(
                question=question,
                num_examples=config.NUM_EXAMPLES,
                include_steps=True
            )

            model_answer = model_result.get('answer', '')
            model_time = time.time() - start_time

            logger.info(f"✓ Model answered in {model_time:.2f}s")

            # STEP 2: Validate and improve with Gemini (if enabled)
            final_answer = model_answer
            answer_source = 'finetuned_model'
            validation_result = None

            if gemini_validator and model_answer:
                try:
                    logger.info("Step 2: Validating with Gemini...")
                    validation_start = time.time()

                    validation_result = gemini_validator.validate_and_improve(
                        question=question,
                        model_answer=model_answer,
                        include_steps=True
                    )

                    validation_time = time.time() - validation_start
                    logger.info(f"✓ Validation completed in {validation_time:.2f}s")

                    # Check if Gemini actually provided a valid answer
                    if validation_result['improved_answer'] and validation_result['confidence'] > 0:
                        # Use Gemini's improved version
                        final_answer = validation_result['improved_answer']
                        answer_source = 'gemini_improved'
                    else:
                        # Gemini failed, use model answer
                        logger.warning("Gemini returned invalid answer, using model answer")
                        final_answer = model_answer
                        answer_source = 'finetuned_model_fallback'
                        validation_result = None

                    # Log for training data collection (only if validation succeeded)
                    if validation_result and validation_result['confidence'] > 0:
                        ValidationLogger.log_validation(
                            question=question,
                            model_answer=model_answer,
                            gemini_result=validation_result,
                            student_id=student_id
                        )

                    logger.info(f"Answer source: {answer_source}")

                except Exception as e:
                    logger.error(f"Gemini validation failed: {e}")
                    # Fall back to model answer if Gemini fails
                    final_answer = model_answer
                    answer_source = 'finetuned_model_fallback'
                    validation_result = None

            else:
                logger.info("Gemini validation disabled, using model answer")

            # STEP 3: Prepare final response
            total_time_ms = int((time.time() - start_time) * 1000)

            response = {
                'status': 'success',
                'answer': final_answer,
                'source': answer_source,
                'response_time_ms': total_time_ms,
                'student_id': student_id,
                'model_used': model_result.get('model_used', 'unknown'),

                # Additional metadata
                'validation': {
                    'enabled': gemini_validator is not None,
                    'is_correct': validation_result['is_correct'] if validation_result else None,
                    'confidence': validation_result['confidence'] if validation_result else None,
                    'final_answer': validation_result.get('final_answer', '') if validation_result else ''
                } if validation_result else {'enabled': False},

                # Include similar problems for reference
                'similar_problems_count': model_result.get('num_retrieved', 0)
            }

            logger.info(f"✓ Complete response in {total_time_ms}ms")

            return jsonify(response)

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
    print("MATH RAG API SERVER - FINE-TUNED + GEMINI VALIDATION")
    print("="*70)
    print("\n⚠️  First startup takes 2-3 minutes to load model")
    print("   After that, responses are MUCH faster!")
    print(f"\n📊 Configuration:")
    print(f"   Max generation time: {config.GENERATION_TIMEOUT}s")
    print(f"   Max output length: {config.MAX_LENGTH} tokens")
    print(f"   RAG examples: {config.NUM_EXAMPLES}")
    print(f"   Gemini validation: {'✓ ENABLED' if config.GEMINI_VALIDATION_ENABLED and config.GEMINI_API_KEY else '✗ DISABLED'}")
    print(f"\n🌐 Starting on http://{config.API_HOST}:{config.API_PORT}")
    print("\n💡 Flow: Fine-tuned Model → Gemini Validation → Best Answer")
    print("="*70 + "\n")

    app.run(
        host=config.API_HOST,
        port=config.API_PORT,
        debug=False,
        threaded=True
    )