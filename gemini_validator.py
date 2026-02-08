"""
Gemini API Validator for Math Answers
Validates and improves model responses using Google Gemini
"""

import google.generativeai as genai
import json
import logging
from typing import Dict, Optional
from datetime import datetime

import config

# Setup logging
logger = logging.getLogger(__name__)

class GeminiValidator:
    """Validates and improves math answers using Gemini API"""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Gemini validator"""

        self.api_key = api_key or config.GEMINI_API_KEY

        if not self.api_key:
            raise ValueError("Gemini API key not provided. Set GEMINI_API_KEY in config.py")

        # Configure Gemini
        genai.configure(api_key=self.api_key)

        # Initialize model
        self.model = genai.GenerativeModel(
            model_name=config.GEMINI_MODEL,
            generation_config={
                "temperature": config.GEMINI_TEMPERATURE,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 2048,
            }
        )

        logger.info(f"✓ Gemini validator initialized with {config.GEMINI_MODEL}")

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

        logger.info("Validating answer with Gemini...")

        # Create validation prompt
        prompt = self._create_validation_prompt(question, model_answer, include_steps)

        try:
            # Call Gemini API
            response = self.model.generate_content(prompt)

            # Parse response
            result = self._parse_gemini_response(response.text)

            logger.info(f"Validation result: {'✓ Correct' if result['is_correct'] else '✗ Incorrect'}")

            return result

        except Exception as e:
            logger.error(f"Gemini validation failed: {e}")

            # Fallback: treat as incorrect and generate new answer
            return self._generate_fallback_answer(question, include_steps)

    def _create_validation_prompt(
        self,
        question: str,
        model_answer: str,
        include_steps: bool
    ) -> str:
        """Create prompt for Gemini validation"""

        prompt = f"""ඔබ ගණිත ගුරුවරයෙකු වන අතර සිංහල භාෂාවෙන් ගණිත ගැටළු විසඳන්නෙකි.

පහත ගණිත ප්‍රශ්නය සහ ලබා දී ඇති පිළිතුර පරීක්ෂා කරන්න:

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
  "is_correct": true/false,
  "confidence": 0.0-1.0,
  "improved_answer": "පියවරෙන් පියවර සිංහල විසඳුම",
  "final_answer": "අවසාන සංඛ්‍යාත්මක පිළිතුර",
  "validation_notes": "පැහැදිලි කිරීම් (optional)"
}}
```

**වැදගත්:**
- විසඳුම සම්පූර්ණයෙන් සිංහල භාෂාවෙන් විය යුතුය
- සෑම පියවරක්ම පැහැදිලිව පැහැදිලි කරන්න
- ගණිතමය සංකේත නිවැරදිව භාවිතා කරන්න
- පිළිතුර සරලව හා තේරුම් ගත හැකි විය යුතුය
"""

        return prompt

    def _parse_gemini_response(self, response_text: str) -> Dict:
        """Parse Gemini's JSON response"""

        try:
            # Extract JSON from response (Gemini sometimes wraps it in markdown)
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            else:
                json_text = response_text.strip()

            # Parse JSON
            result = json.loads(json_text)

            # Validate required fields
            required_fields = ['is_correct', 'confidence', 'improved_answer', 'final_answer']
            for field in required_fields:
                if field not in result:
                    raise ValueError(f"Missing required field: {field}")

            return result

        except Exception as e:
            logger.error(f"Failed to parse Gemini response: {e}")
            logger.debug(f"Raw response: {response_text}")

            # Return safe fallback
            return {
                'is_correct': False,
                'confidence': 0.0,
                'improved_answer': response_text,  # Use raw response
                'final_answer': '',
                'validation_notes': f'Parsing error: {str(e)}'
            }

    def _generate_fallback_answer(self, question: str, include_steps: bool) -> Dict:
        """Generate answer directly with Gemini when validation fails"""

        logger.info("Generating fallback answer with Gemini...")

        prompt = f"""ප්‍රශ්නය: {question}

මෙම ගණිත ගැටලුව පියවරෙන් පියවර සිංහලෙන් විසඳන්න.

විසඳුම:
"""

        try:
            response = self.model.generate_content(prompt)

            return {
                'is_correct': True,  # Assume Gemini is correct
                'confidence': 0.9,
                'improved_answer': response.text,
                'final_answer': self._extract_final_answer(response.text),
                'validation_notes': 'Generated by Gemini (validation failed)'
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

        # Look for common patterns
        patterns = [
            'අවසාන පිළිතුර:',
            'පිළිතුර:',
            'උත්තරය:',
            'Final answer:',
            'Answer:'
        ]

        for pattern in patterns:
            if pattern in text:
                # Get text after pattern
                parts = text.split(pattern)
                if len(parts) > 1:
                    answer = parts[-1].strip()
                    # Get first line or sentence
                    answer = answer.split('\n')[0].strip()
                    return answer

        # If no pattern found, return last line
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
        gemini_result: Dict,
        student_id: Optional[str] = None
    ):
        """Save validation data for future fine-tuning"""

        timestamp = datetime.now().isoformat()

        log_entry = {
            'timestamp': timestamp,
            'student_id': student_id,
            'question': question,
            'model_answer': model_answer,
            'gemini_validation': gemini_result,
            'used_gemini': not gemini_result['is_correct']
        }

        # Save to file
        log_file = f"{config.VALIDATION_LOGS_DIR}/{datetime.now().strftime('%Y%m%d')}_validations.jsonl"

        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

            logger.debug(f"Validation logged to {log_file}")

        except Exception as e:
            logger.error(f"Failed to log validation: {e}")


# Test function
if __name__ == "__main__":
    # Test the validator
    validator = GeminiValidator()

    test_question = "x + 5 = 10 නම් x හි අගය කීයද?"
    test_answer = "x = 5"

    result = validator.validate_and_improve(test_question, test_answer)

    print("\n" + "="*70)
    print("VALIDATION RESULT")
    print("="*70)
    print(f"Is Correct: {result['is_correct']}")
    print(f"Confidence: {result['confidence']}")
    print(f"\nImproved Answer:\n{result['improved_answer']}")
    print(f"\nFinal Answer: {result['final_answer']}")
    print("="*70)
