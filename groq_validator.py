"""
Groq API Validator for Math Answers
Validates and improves model responses using Groq (Llama 3.3 70B)
"""

from groq import Groq
import json
import logging
from typing import Dict, Optional
from datetime import datetime
import sys

import config

# Setup logging with UTF-8 encoding 
if sys.platform == 'win32':
    # Fix Unicode encoding errors on Windows console
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

logger = logging.getLogger(__name__)


class GroqValidator:
    """Validates and improves math answers using Groq API"""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Groq validator"""

        self.api_key = api_key or config.GROQ_API_KEY

        if not self.api_key:
            raise ValueError("Groq API key not provided. Set GROQ_API_KEY in config.py")

        # Initialize Groq client
        self.client = Groq(api_key=self.api_key)
        self.model = config.GROQ_MODEL

        logger.info(f"✓ Groq validator initialized with {self.model}")

    def validate_and_improve(
        self,
        question: str,
        model_answer: str,
        include_steps: bool = True
    ) -> Dict:
        """
        Validate model answer and provide improved version

        Returns:
            {
                'is_correct': bool,
                'confidence': float (0-1),
                'improved_answer': str (Sinhala step-by-step solution),
                'final_answer': str,
                'validation_notes': str
            }
        """

        logger.info("Validating answer with Groq...")

        # Create validation prompt
        prompt = self._create_validation_prompt(question, model_answer, include_steps)

        try:
            # Call Groq API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """ඔබ ගණිත ගුරුවරයෙකු වන අතර සිංහල භාෂාවෙන් ගණිත ගැටළු විසඳන්නෙකි.

You are a math teacher who solves problems step by step in Sinhala language.

**Important Rules:**
- ONLY answer mathematics questions
- Supported topics: ලඝුගණක (logarithms), ශ්‍රීඝ්‍රතාවය (velocity), සමාන්තර ශ්‍රේණි (arithmetic progression), equations
- Do NOT answer: history, science, general knowledge, chitchat
- Do NOT answer: questions requiring diagrams, images, shapes, graphs, or visual elements
- If question mentions රූප/diagram/චිත්‍ර/image/හැඩ/shape/ප්‍රස්තාර/graph, decline politely
- If question is not about math or requires visuals, set is_correct=false with validation note explaining why
- Always respond in valid JSON format"""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=config.GROQ_TEMPERATURE,
                max_tokens=2048
            )

            # Parse response
            response_text = response.choices[0].message.content
            result = self._parse_response(response_text)

            logger.info(f"Validation result: {'✓ Correct' if result['is_correct'] else '✗ Incorrect'}")

            return result

        except Exception as e:
            logger.error(f"Groq validation failed: {e}")

            # Fallback: generate new answer
            return self._generate_fallback_answer(question, include_steps)

    def _create_validation_prompt(
        self,
        question: str,
        model_answer: str,
        include_steps: bool
    ) -> str:
        """Create prompt for Groq validation"""

        prompt = f"""පහත ගණිත ප්‍රශ්නය සහ ලබා දී ඇති පිළිතුර පරීක්ෂා කරන්න:

**ප්‍රශ්නය:**
{question}

**ලබා දී ඇති පිළිතුර:**
{model_answer}

**ඔබේ කාර්යය:**
1. ලබා දී ඇති පිළිතුර නිවැරදිද නැද්ද පරීක්ෂා කරන්න
2. පියවරෙන් පියවර නිවැරදි විසඳුම සිංහලෙන් ලබා දෙන්න
3. අවසාන පිළිතුර ලබා දෙන්න

**ප්‍රතිදානය මෙලෙස JSON ආකෘතියෙන් ලබා දෙන්න:**
```json
{{
  "is_correct": true හෝ false,
  "confidence": 0.0 සිට 1.0 දක්වා අගයක්,
  "improved_answer": "පියවරෙන් පියවර සිංහල විසඳුම මෙහි ලියන්න",
  "final_answer": "අවසාන සංඛ්‍යාත්මක පිළිතුර",
  "validation_notes": "අමතර සටහන් (optional)"
}}
```

**වැදගත්:**
- විසඳුම සම්පූර්ණයෙන් සිංහල භාෂාවෙන් විය යුතුය
- සෑම පියවරක්ම පැහැදිලිව පැහැදිලි කරන්න
- ගණිතමය සංකේත නිවැරදිව භාවිතා කරන්න
- JSON format එක නිවැරදිව ලබා දෙන්න"""

        return prompt

    def _parse_response(self, response_text: str) -> Dict:
        """Parse Groq's JSON response"""

        try:
            # Extract JSON from response (might be wrapped in markdown)
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            elif "{" in response_text:
                # Find JSON object in text
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                json_text = response_text[json_start:json_end]
            else:
                json_text = response_text.strip()

            # Parse JSON (use strict=False to allow control characters in strings)
            result = json.loads(json_text, strict=False)

            # Validate required fields
            required_fields = ['is_correct', 'confidence', 'improved_answer', 'final_answer']
            for field in required_fields:
                if field not in result:
                    result[field] = "" if field in ['improved_answer', 'final_answer'] else False if field == 'is_correct' else 0.0

            # Ensure types are correct
            result['is_correct'] = bool(result.get('is_correct', False))
            result['confidence'] = float(result.get('confidence', 0.0))
            result['improved_answer'] = str(result.get('improved_answer', ''))
            result['final_answer'] = str(result.get('final_answer', ''))
            result['validation_notes'] = str(result.get('validation_notes', ''))

            return result

        except Exception as e:
            logger.error(f"Failed to parse Groq response: {e}")
            logger.debug(f"Raw response: {response_text}")

            # If JSON parsing fails but we have text, use it as the answer
            if response_text and len(response_text) > 50:
                return {
                    'is_correct': True,
                    'confidence': 0.7,
                    'improved_answer': response_text,
                    'final_answer': '',
                    'validation_notes': 'JSON parsing failed, using raw response'
                }

            return {
                'is_correct': False,
                'confidence': 0.0,
                'improved_answer': '',
                'final_answer': '',
                'validation_notes': f'Parsing error: {str(e)}'
            }

    def _generate_fallback_answer(self, question: str, include_steps: bool) -> Dict:
        """Generate answer directly with Groq when validation fails"""

        logger.info("Generating fallback answer with Groq...")

        prompt = f"""ප්‍රශ්නය: {question}

මෙම ගණිත ගැටලුව පියවරෙන් පියවර සිංහලෙන් විසඳන්න.

විසඳුම:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "ඔබ ගණිත ගුරුවරයෙකි. සිංහල භාෂාවෙන් පියවරෙන් පියවර ගණිත ගැටළු විසඳන්න. ONLY answer mathematics questions. Do NOT answer questions requiring diagrams, images, shapes, or graphs. If the question is not about math or requires visuals, politely decline."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=config.GROQ_TEMPERATURE,
                max_tokens=2048
            )

            answer_text = response.choices[0].message.content

            return {
                'is_correct': True,
                'confidence': 0.9,
                'improved_answer': answer_text,
                'final_answer': self._extract_final_answer(answer_text),
                'validation_notes': 'Generated directly by Groq'
            }

        except Exception as e:
            logger.error(f"Fallback generation failed: {e}")

            return {
                'is_correct': False,
                'confidence': 0.0,
                'improved_answer': 'ගැටලුවක් ඇති විය. කරුණාකර නැවත උත්සාහ කරන්න.',
                'final_answer': '',
                'validation_notes': f'Error: {str(e)}'
            }

    def _extract_final_answer(self, text: str) -> str:
        """Extract final answer from solution text"""

        patterns = [
            'අවසාන පිළිතුර:',
            'පිළිතුර:',
            'උත්තරය:',
            'Final answer:',
            'Answer:',
            '∴'
        ]

        for pattern in patterns:
            if pattern in text:
                parts = text.split(pattern)
                if len(parts) > 1:
                    answer = parts[-1].strip()
                    answer = answer.split('\n')[0].strip()
                    return answer

        lines = text.strip().split('\n')
        return lines[-1].strip() if lines else ''


# ============================================================================
# VALIDATION LOGGER
# ============================================================================

class ValidationLogger:
    """Log validation results for future training data"""

    @staticmethod
    def log_validation(
        question: str,
        model_answer: str,
        groq_result: Dict,
        student_id: Optional[str] = None
    ):
        """Save validation data for future fine-tuning"""

        timestamp = datetime.now().isoformat()

        log_entry = {
            'timestamp': timestamp,
            'student_id': student_id,
            'question': question,
            'model_answer': model_answer,
            'groq_validation': groq_result,
            'used_groq': not groq_result['is_correct']
        }

        log_file = f"{config.VALIDATION_LOGS_DIR}/{datetime.now().strftime('%Y%m%d')}_validations.jsonl"

        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

            logger.debug(f"Validation logged to {log_file}")

        except Exception as e:
            logger.error(f"Failed to log validation: {e}")


# Backwards compatibility aliases
GeminiValidator = GroqValidator


# Test function
if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING GROQ VALIDATOR")
    print("="*70)

    try:
        validator = GroqValidator()

        test_question = "x + 5 = 10 නම් x හි අගය කීයද?"
        test_answer = "x = 5"

        print(f"\nQuestion: {test_question}")
        print(f"Model Answer: {test_answer}")
        print("\nValidating...")

        result = validator.validate_and_improve(test_question, test_answer)

        print("\n" + "="*70)
        print("VALIDATION RESULT")
        print("="*70)
        print(f"Is Correct: {result['is_correct']}")
        print(f"Confidence: {result['confidence']}")
        print(f"\nImproved Answer:\n{result['improved_answer']}")
        print(f"\nFinal Answer: {result['final_answer']}")
        print("="*70)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
