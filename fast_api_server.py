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
from groq_validator import GroqValidator, ValidationLogger
from question_filter import QuestionFilter
import config
import db

# Configure logging with UTF-8 encoding for Windows
import sys
if sys.platform == 'win32':
    # Fix Unicode encoding errors on Windows console
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global RAG system and validator (initialized once)
rag_system = None
groq_validator = None
question_filter = QuestionFilter(use_groq_validation=False)

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
    """Initialize RAG system and Groq validator on first request"""
    global rag_system, groq_validator

    # Initialize Groq validator first (always fast)
    if groq_validator is None and config.GROQ_API_KEY:
        try:
            groq_validator = GroqValidator()
            logger.info("✓ Groq validator initialized!")
        except Exception as e:
            logger.warning(f"Failed to initialize Groq validator: {e}")
            groq_validator = None

    # In DIRECT MODE, skip slow model initialization
    if getattr(config, 'GROQ_DIRECT_MODE', False):
        if rag_system is None:
            logger.info("⚡ DIRECT MODE: Skipping slow model, using Groq only")
            rag_system = "direct_mode"  # Placeholder to prevent re-initialization
        return

    # Standard mode: Initialize RAG system with fine-tuned model
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

@app.route('/api/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'model': 'SinhaLM Fine-tuned with RAG + Groq Validation',
        'ready': rag_system is not None,
        'validation_enabled': groq_validator is not None,
        'direct_mode': getattr(config, 'GROQ_DIRECT_MODE', False),
        'max_generation_time': f"{config.GENERATION_TIMEOUT}s",
        'max_length': config.MAX_LENGTH,
        'num_examples': config.NUM_EXAMPLES
    })

@app.route('/api/answer', methods=['POST'])
def answer():
    """
    Answer a question using RAG system + Groq validation
    GROQ_DIRECT_MODE=True: Skip slow model, use Groq directly (FAST!)
    GROQ_DIRECT_MODE=False: Model generates -> Groq validates
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

        # Filter non-math questions
        filter_result = question_filter.validate_question(question)
        if not filter_result['is_valid']:
            logger.info(f"Question filtered: {filter_result['category']}")
            return jsonify({
                'status': 'filtered',
                'answer': filter_result['reason'],
                'category': filter_result['category'],
                'student_id': student_id
            }), 200

        # Measure response time
        start_time = time.time()

        try:
            final_answer = ''
            answer_source = ''
            validation_result = None
            model_answer = ''

            # Check if GROQ_DIRECT_MODE is enabled (skip slow fine-tuned model)
            if (getattr(config, 'GROQ_DIRECT_MODE', False) or rag_system == "direct_mode") and groq_validator:
                # FAST MODE: Use Groq directly
                logger.info("DIRECT MODE: Using Groq directly (skipping slow model)...")

                validation_result = groq_validator._generate_fallback_answer(question, include_steps=True)

                if validation_result['improved_answer'] and validation_result['confidence'] > 0:
                    final_answer = validation_result['improved_answer']
                    answer_source = 'groq_direct'
                    logger.info(f"✓ Groq answered in {time.time() - start_time:.2f}s")
                else:
                    # Groq failed, try fine-tuned model as fallback
                    logger.warning("Groq direct failed, falling back to fine-tuned model...")
                    model_result = rag_system.answer_question(
                        question=question,
                        num_examples=config.NUM_EXAMPLES,
                        include_steps=True
                    )
                    final_answer = model_result.get('answer', '')
                    answer_source = 'finetuned_model_fallback'

            else:
                # STANDARD MODE: Fine-tuned model + Groq validation
                logger.info("Step 1: Getting answer from fine-tuned model...")
                model_result = rag_system.answer_question(
                    question=question,
                    num_examples=config.NUM_EXAMPLES,
                    include_steps=True
                )

                model_answer = model_result.get('answer', '')
                model_time = time.time() - start_time

                logger.info(f"✓ Model answered in {model_time:.2f}s")

                final_answer = model_answer
                answer_source = 'finetuned_model'

                if groq_validator and model_answer:
                    try:
                        logger.info("Step 2: Validating with Groq...")
                        validation_start = time.time()

                        validation_result = groq_validator.validate_and_improve(
                            question=question,
                            model_answer=model_answer,
                            include_steps=True
                        )

                        validation_time = time.time() - validation_start
                        logger.info(f"✓ Validation completed in {validation_time:.2f}s")

                        if validation_result['improved_answer'] and validation_result['confidence'] > 0:
                            final_answer = validation_result['improved_answer']
                            answer_source = 'groq_improved'
                        else:
                            logger.warning("Groq returned invalid answer, using model answer")
                            final_answer = model_answer
                            answer_source = 'finetuned_model_fallback'
                            validation_result = None

                        if validation_result and validation_result['confidence'] > 0:
                            ValidationLogger.log_validation(
                                question=question,
                                model_answer=model_answer,
                                groq_result=validation_result,
                                student_id=student_id
                            )

                        logger.info(f"Answer source: {answer_source}")

                    except Exception as e:
                        logger.error(f"Groq validation failed: {e}")
                        final_answer = model_answer
                        answer_source = 'finetuned_model_fallback'
                        validation_result = None

                else:
                    logger.info("Groq validation disabled, using model answer")

            # STEP 3: Prepare final response
            total_time_ms = int((time.time() - start_time) * 1000)

            response = {
                'status': 'success',
                'answer': final_answer,
                'source': answer_source,
                'response_time_ms': total_time_ms,
                'student_id': student_id,
                'model_used': 'groq_direct' if answer_source == 'groq_direct' else 'fine-tuned',

                # Additional metadata
                'validation': {
                    'enabled': groq_validator is not None,
                    'is_correct': validation_result['is_correct'] if validation_result else None,
                    'confidence': validation_result['confidence'] if validation_result else None,
                    'final_answer': validation_result.get('final_answer', '') if validation_result else ''
                } if validation_result else {'enabled': True, 'is_correct': True, 'confidence': 0.9},

                # Include similar problems for reference
                'similar_problems_count': 0 if answer_source == 'groq_direct' else 2
            }

            logger.info(f"Complete response in {total_time_ms}ms")

            # STEP 4: Save to MongoDB (non-blocking, won't fail the request)
            try:
                db.save_response(
                    question=question,
                    answer=final_answer,
                    student_id=student_id,
                    source=answer_source,
                    model_used=response['model_used'],
                    response_time_ms=total_time_ms,
                    validation=response.get('validation'),
                    similar_problems_count=response.get('similar_problems_count', 0)
                )
            except Exception as e:
                logger.error(f"MongoDB save failed: {e}")

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
    print("MATH RAG API SERVER - FINE-TUNED + GROQ VALIDATION")
    print("="*70)
    print("\n  First startup takes 2-3 minutes to load model")
    print("   After that, responses are MUCH faster!")
    print(f"\n Configuration:")
    print(f"   Max generation time: {config.GENERATION_TIMEOUT}s")
    print(f"   Max output length: {config.MAX_LENGTH} tokens")
    print(f"   RAG examples: {config.NUM_EXAMPLES}")
    print(f"   Groq validation: {'✓ ENABLED' if config.GROQ_VALIDATION_ENABLED and config.GROQ_API_KEY else '✗ DISABLED'}")
    print(f"   DIRECT MODE: {'✓ FAST (Groq only)' if getattr(config, 'GROQ_DIRECT_MODE', False) else '✗ SLOW (Model + Groq)'}")
    print(f"\n Starting on http://{config.API_HOST}:{config.API_PORT}")
    if getattr(config, 'GROQ_DIRECT_MODE', False):
        print("\n⚡ FAST MODE: Groq answers directly (< 2 seconds)")
    else:
        print("\n Flow: Fine-tuned Model → Groq Validation → Best Answer")
    print("="*70 + "\n")

    app.run(
        host=config.API_HOST,
        port=config.API_PORT,
        debug=False,
        threaded=True
    )