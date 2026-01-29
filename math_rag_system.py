"""
Complete Math RAG System - Core Implementation
Supports both original SinhaLM and fine-tuned models
"""

import json
import os
import torch
import faiss
import numpy as np
import logging
from typing import List, Dict, Optional
from datetime import datetime
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

import config

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# DATA PROCESSOR
# ============================================================================

class MathProblemProcessor:
    """Process mathematics problems from JSON"""
    
    @staticmethod
    def load_problems(json_file: str) -> List[Dict]:
        """Load problems from JSON"""
        logger.info(f"Loading problems from {json_file}")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both old format (with 'examples' wrapper) and new format (direct array)
        if isinstance(data, dict) and 'examples' in data:
            problems = data['examples']
        elif isinstance(data, list):
            problems = data
        else:
            raise ValueError("Unexpected JSON format")
        
        logger.info(f"Loaded {len(problems)} problems")
        
        return problems
    
    @staticmethod
    def format_problem(problem: Dict) -> Dict:
        """Format a single problem - handles multiple formats"""
        
        # Detect format type
        has_question_id = 'Question_ID' in problem
        has_question_text = 'Question_Text' in problem
        has_capital_question = 'Question' in problem
        has_lowercase_question = 'question' in problem
        has_text_field = 'text' in problem
        has_metadata = 'metadata' in problem
        
        # FORMAT 1: NEW STANDARD (Question_ID + Question_Text)
        if has_question_id or has_question_text:
            solution_methods = problem.get('Solution_Methods', [])
            
            solution = ""
            formatted_steps = []
            solution_method = ""
            is_simplest = True
            
            if solution_methods and len(solution_methods) > 0:
                first_method = solution_methods[0]
                steps = first_method.get('Steps', [])
                solution_method = first_method.get('Method_Name', '')
                is_simplest = first_method.get('Is_Simplest', True)
                
                for step in steps:
                    step_num = step.get('Step_Number', '')
                    step_text = step.get('Step_Text', '')
                    step_explanation = step.get('Step_Explanation', '')
                    
                    if step_explanation:
                        solution += f"{step_explanation}\n"
                    if step_text:
                        solution += f"{step_text}\n"
                    
                    formatted_steps.append({
                        'step_number': step_num,
                        'step_description': step_explanation,
                        'step_answer': step_text
                    })
            
            topic = problem.get('Lesson_Topic', '')
            sub_topic = problem.get('Sub_Topic', '')
            
            return {
                'id': problem.get('Question_ID', ''),
                'question': problem.get('Question_Text', ''),
                'topic': topic,
                'sub_topic': sub_topic,
                'difficulty': problem.get('Difficulty', ''),
                'solution': solution.strip(),
                'final_answer': problem.get('Final_Answer', ''),
                'type': 'exercises',
                'steps': formatted_steps,
                'solution_method': solution_method,
                'is_simplest': is_simplest,
                'search_text': f"{topic} {sub_topic} {problem.get('Question_Text', '')}"
            }
        
        # FORMAT 2: EXTRACTED EXERCISES (text + metadata with sub_questions)
        elif has_text_field and has_metadata:
            metadata = problem.get('metadata', {})
            main_question = metadata.get('main_question', problem.get('text', ''))
            sub_questions = metadata.get('sub_questions', [])
            
            # Combine main question with sub-questions
            full_question = main_question
            if sub_questions:
                full_question += "\n" + "\n".join([sq.get('question', '') for sq in sub_questions])
            
            solution = ""
            formatted_steps = []
            
            # Extract solutions from sub_questions if they have answers
            for i, sq in enumerate(sub_questions, 1):
                if 'solution' in sq or 'answer' in sq:
                    step_text = sq.get('solution', sq.get('answer', ''))
                    solution += f"{i}. {step_text}\n"
                    formatted_steps.append({
                        'step_number': i,
                        'step_description': sq.get('question', ''),
                        'step_answer': step_text
                    })
            
            return {
                'id': '',
                'question': full_question,
                'topic': problem.get('topic', 'සංඛ්‍යා'),
                'sub_topic': problem.get('type', ''),
                'difficulty': '',
                'solution': solution.strip(),
                'final_answer': problem.get('final_answer', ''),
                'type': problem.get('type', 'exercise'),
                'steps': formatted_steps,
                'solution_method': '',
                'is_simplest': True,
                'search_text': f"{problem.get('topic', '')} {full_question}"
            }
        
        # FORMAT 3: CAPITAL LETTER FORMAT (Question + Steps + Final_answer)
        elif has_capital_question:
            solution = ""
            formatted_steps = []
            
            for step_data in problem.get('Steps', []):
                step_text = step_data.get('Step', '')
                solution += f"{step_text}\n"
                
                formatted_steps.append({
                    'step_number': len(formatted_steps) + 1,
                    'step_description': '',
                    'step_answer': step_text
                })
            
            return {
                'id': '',
                'question': problem.get('Question', ''),
                'topic': problem.get('topic', 'සංඛ්‍යා'),
                'sub_topic': problem.get('sub_topic', ''),
                'difficulty': '',
                'solution': solution.strip(),
                'final_answer': problem.get('Final_answer', ''),
                'type': problem.get('type', 'exercises'),
                'steps': formatted_steps,
                'solution_method': '',
                'is_simplest': True,
                'search_text': f"{problem.get('topic', '')} {problem.get('Question', '')}"
            }
        
        # FORMAT 4: OLD LOWERCASE FORMAT (question + steps)
        elif has_lowercase_question:
            solution = ""
            formatted_steps = []
            
            for step in problem.get('steps', []):
                if step.get('step_description'):
                    solution += step['step_description'] + "\n"
                if step.get('step_answer'):
                    solution += step['step_answer'] + "\n"
                
                formatted_steps.append(step)
            
            return {
                'id': '',
                'question': problem.get('question', ''),
                'topic': problem.get('topic', ''),
                'sub_topic': problem.get('sub_topic', ''),
                'difficulty': '',
                'solution': solution.strip(),
                'final_answer': problem.get('final_answer', ''),
                'type': problem.get('type', 'exercises'),
                'steps': formatted_steps,
                'solution_method': '',
                'is_simplest': True,
                'search_text': f"{problem.get('topic', '')} {problem.get('sub_topic', '')} {problem.get('question', '')}"
            }
        
        else:
            # UNKNOWN FORMAT - Show helpful error
            raise ValueError(f"Unknown problem format. Keys found: {list(problem.keys())}")
        
    @staticmethod
    def process_all(json_file: str) -> List[Dict]:
        """Load and format all problems"""
        raw_problems = MathProblemProcessor.load_problems(json_file)
        formatted = []

        for p in raw_problems:
            try:
                formatted.append(MathProblemProcessor.format_problem(p))
            except ValueError as e:
                logger.warning(f"Skipping unknown format: {e}")

        logger.info(f"Processed {len(formatted)} problems (skipped {len(raw_problems) - len(formatted)})")
        return formatted

    
    @staticmethod
    def get_statistics(problems: List[Dict]) -> Dict:
        """Get statistics about problems"""
        stats = {
            'total': len(problems),
            'topics': {},
            'sub_topics': {},
            'difficulty': {}
        }
        
        for p in problems:
            topic = p['topic']
            sub_topic = p['sub_topic']
            difficulty = p.get('difficulty', '')
            
            stats['topics'][topic] = stats['topics'].get(topic, 0) + 1
            
            if sub_topic:
                stats['sub_topics'][sub_topic] = stats['sub_topics'].get(sub_topic, 0) + 1
            
            if difficulty:
                stats['difficulty'][difficulty] = stats['difficulty'].get(difficulty, 0) + 1
        
        return stats

# ============================================================================
# RAG DATABASE
# ============================================================================

class MathRAGDatabase:
    """Vector database for mathematics problems"""
    
    def __init__(self, embedding_model: str = config.EMBEDDING_MODEL):
        logger.info(f"Initializing RAG database with {embedding_model}")
        
        self.embedder = SentenceTransformer(embedding_model)
        self.index = None
        self.problems = []
        
    def build_index(self, problems: List[Dict], batch_size: int = config.EMBEDDING_BATCH_SIZE):
        """Build FAISS index"""
        logger.info(f"Building index for {len(problems)} problems")
        
        self.problems = problems
        
        # Create embeddings
        search_texts = [p['search_text'] for p in problems]
        logger.info("Generating embeddings...")
        
        embeddings = self.embedder.encode(
            search_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        
        logger.info("✓ Index built successfully")
    
    def search(self, query: str, k: int = config.NUM_EXAMPLES) -> List[Dict]:
        """Search for similar problems"""
        
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        query_emb = self.embedder.encode([query], convert_to_numpy=True)
        
        distances, indices = self.index.search(
            query_emb.astype('float32'),
            min(k, len(self.problems))
        )
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            results.append({
                'problem': self.problems[idx],
                'distance': float(dist),
                'similarity': float(1 / (1 + dist))
            })
        
        return results
    
    def save(self, path: str = config.RAG_INDEX_DIR):
        """Save index and problems"""
        logger.info(f"Saving index to {path}")
        
        os.makedirs(path, exist_ok=True)
        
        faiss.write_index(self.index, os.path.join(path, "faiss.index"))
        
        with open(os.path.join(path, "problems.json"), 'w', encoding='utf-8') as f:
            json.dump(self.problems, f, ensure_ascii=False, indent=2)
        
        logger.info("✓ Index saved")
    
    def load(self, path: str = config.RAG_INDEX_DIR):
        """Load saved index"""
        logger.info(f"Loading index from {path}")
        
        self.index = faiss.read_index(os.path.join(path, "faiss.index"))
        
        with open(os.path.join(path, "problems.json"), 'r', encoding='utf-8') as f:
            self.problems = json.load(f)
        
        logger.info(f"✓ Index loaded ({len(self.problems)} problems)")

# ============================================================================
# MAIN RAG SYSTEM
# ============================================================================

class SinhaLMMathRAG:
    """Complete RAG system with SinhaLM"""
    
    def __init__(
        self, 
        rag_database: MathRAGDatabase, 
        model_path: str = config.SINHALM_MODEL_PATH,
        use_finetuned: bool = False
    ):
        logger.info("="*70)
        logger.info("INITIALIZING SINHALM MATH RAG SYSTEM")
        logger.info("="*70)
        
        self.rag_db = rag_database
        self.use_finetuned = use_finetuned
        
        # Determine which model to use
        if use_finetuned and os.path.exists(config.FINETUNED_MODEL_PATH):
            logger.info("Using FINE-TUNED model")
            model_path = config.FINETUNED_MODEL_PATH
        else:
            logger.info("Using ORIGINAL SinhaLM model")
            model_path = config.SINHALM_MODEL_PATH
        
        # Load model
        self._load_model(model_path)
        
        logger.info("✓ System ready!")
    
    def _load_model(self, model_path: str):
        """Load SinhaLM model"""
        
        logger.info(f"Loading tokenizer from {config.BASE_MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Loading base model: {config.BASE_MODEL_NAME}")
        base_model = AutoModelForCausalLM.from_pretrained(
            config.BASE_MODEL_NAME,
            device_map=config.DEVICE,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
        logger.info(f"Loading adapter from {model_path}")
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.model.eval()
        
        logger.info("✓ Model loaded successfully")
    
    def answer_question(
        self,
        question: str,
        num_examples: int = config.NUM_EXAMPLES,
        include_steps: bool = True
    ) -> Dict:
        """Answer a mathematics question"""
        
        logger.info(f"Processing question: {question[:50]}...")
        
        # Search similar problems
        similar = self.rag_db.search(question, k=num_examples)
        
        if not similar:
            logger.warning("No similar problems found")
            return {
                'status': 'no_results',
                'answer': 'සමාන ගැටළු සොයා ගත නොහැකි විය.',
                'similar_problems': []
            }
        
        # Build context
        context = self._build_context(similar, include_steps)
        
        # Create prompt
        prompt = self._create_prompt(question, context, include_steps)
        
        # Generate answer
        answer = self._generate(prompt)
        
        return {
            'status': 'success',
            'answer': answer,
            'similar_problems': similar,
            'num_retrieved': len(similar),
            'model_used': 'fine-tuned' if self.use_finetuned else 'original',
            'timestamp': datetime.now().isoformat()
        }
    
    def _build_context(self, similar_problems: List[Dict], include_steps: bool) -> str:
        """Build context from retrieved problems"""
        
        context = "සමාන ගණිත ගැටළු:\n\n"
        
        for i, result in enumerate(similar_problems, 1):
            problem = result['problem']
            context += f"උදාහරණය {i}:\n"
            context += f"විෂයය: {problem['topic']}"
            
            if problem['sub_topic']:
                context += f" - {problem['sub_topic']}"
            
            context += f"\n\nප්‍රශ්නය: {problem['question']}\n\n"
            
            if include_steps and problem['solution']:
                context += f"විසඳුම:\n{problem['solution']}\n\n"
            
            context += f"අවසාන පිළිතුර: {problem['final_answer']}\n"
            context += "-" * 50 + "\n\n"
        
        return context
    
    # def _create_prompt(self, question: str, context: str, include_steps: bool) -> str:
        """Create generation prompt"""
        
        step_instruction = "පියවරෙන් පියවර විස්තරාත්මක විසඳුමක් සපයන්න." if include_steps else ""
        
        prompt = f"""{context}
දැන්, මෙම නව ගැටලුව විසඳන්න:

ප්‍රශ්නය: {question}

ඉහත උදාහරණ භාවිතා කරමින්, {step_instruction}

විසඳුම:
"""
        return prompt
    
    def _create_prompt(self, question: str, context: str, include_steps: bool) -> str:
        # Detect conceptual vs problem-solving question
        conceptual_keywords = ['ක්‍රම', 'මොනවාද', 'කෙසේ', 'විදිහ', 'ආකාර']
        is_conceptual = any(kw in question for kw in conceptual_keywords)
        
        if is_conceptual:
            # Different prompt for conceptual questions
            prompt = f"""{context}

    මේ උදාහරණ මත පදනම්ව, පහත ප්‍රශ්නයට පැහැදිලි කිරීමක් ලබා දෙන්න:

    ප්‍රශ්නය: {question}

    පැහැදිලි කිරීම:
    """
        else:
            # Original prompt for problem-solving
            step_instruction = "පියවරෙන් පියවර විස්තරාත්මක විසඳුමක් සපයන්න." if include_steps else ""
            prompt = f"""{context}

    දැන්, මෙම නව ගැටලුව විසඳන්න:

    ප්‍රශ්නය: {question}

    ඉහත උදාහරණ භාවිතා කරමින්, {step_instruction}

    විසඳුම:
    """
        
        return prompt

    def _generate(self, prompt: str) -> str:
        """Generate answer using model (CPU optimized with progress)"""
        
        import time
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024  # Reduced context
        )
        
        logger.info("Generating answer...")
        logger.info(f"Input length: {inputs['input_ids'].shape[1]} tokens")
        logger.info(f"Max output: {config.MAX_LENGTH} tokens")
        logger.info(f"Estimated time: 10-30 seconds on CPU")
        
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.MAX_LENGTH,  # Use max_new_tokens instead
                min_new_tokens=50,  # Ensure minimum response
                temperature=config.TEMPERATURE,
                do_sample=config.TEMPERATURE > 0,
                top_p=config.TOP_P,
                top_k=config.TOP_K,
                repetition_penalty=config.REPETITION_PENALTY,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                # CPU optimizations
                use_cache=True,
                early_stopping=True,
                num_beams=1  # Greedy decoding - fastest
            )
        
        generation_time = time.time() - start_time
        logger.info(f"✓ Generation completed in {generation_time:.1f}s")
        
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the answer part
        if "විසඳුම:" in full_response:
            answer = full_response.split("විසඳුම:")[-1].strip()
        else:
            # Remove the prompt from the response
            answer = full_response[len(prompt):].strip()
        
        logger.info(f"✓ Answer length: {len(answer)} characters")
        return answer

# ============================================================================
# SYSTEM BUILDER
# ============================================================================

class SystemBuilder:
    """Build and initialize the complete system"""
    
    @staticmethod
    def build(rebuild_index: bool = False, use_finetuned: bool = False) -> SinhaLMMathRAG:
        """Build the complete RAG system"""
        
        print("\n" + "="*70)
        print("BUILDING MATH RAG SYSTEM")
        print("="*70)
        
        # Validate configuration
        print("\n[1/4] Validating configuration...")
        if not config.validate_config():
            raise ValueError("Configuration validation failed")
        print("✓ Configuration valid")
        
        # Load and process data
        print("\n[2/4] Loading mathematics problems...")
        processor = MathProblemProcessor()
        problems = processor.process_all(config.MATH_PROBLEMS_JSON)
        
        stats = processor.get_statistics(problems)
        print(f"✓ Loaded {stats['total']} problems")
        print(f"  Topics: {len(stats['topics'])}")
        for topic, count in stats['topics'].items():
            print(f"    - {topic}: {count}")
        
        if stats['difficulty']:
            print(f"  Difficulty Levels:")
            for diff, count in stats['difficulty'].items():
                print(f"    - {diff}: {count}")
        
        # Build or load RAG index
        print("\n[3/4] Setting up RAG database...")
        rag_db = MathRAGDatabase()
        
        index_exists = os.path.exists(os.path.join(config.RAG_INDEX_DIR, "faiss.index"))
        
        if rebuild_index or not index_exists:
            print("Building new index...")
            rag_db.build_index(problems)
            rag_db.save()
        else:
            print("Loading existing index...")
            rag_db.load()
        
        print("✓ RAG database ready")
        
        # Initialize RAG system
        print("\n[4/4] Initializing RAG system...")
        rag_system = SinhaLMMathRAG(
            rag_database=rag_db,
            use_finetuned=use_finetuned
        )
        
        print("\n" + "="*70)
        print("✅ SYSTEM BUILD COMPLETE!")
        print("="*70 + "\n")
        
        return rag_system