"""
Question Filter - Validates if questions are math-related and within supported topics
Blocks off-topic or non-math questions before processing
"""

import logging
from typing import Dict, List
import re

import config

logger = logging.getLogger(__name__)


class QuestionFilter:
    """Filter and validate student questions"""

    # Supported topics (in Sinhala)
    SUPPORTED_TOPICS = [
        "ලඝුගණක",           # Logarithms
        "ශ්‍රීඝ්‍රතාවය",       # Velocity/Speed
        "සමාන්තර ශ්‍රේණි",    # Arithmetic Progression
        "වර්ගජ සමීකරණ",      # Quadratic equations
        "රේඛීය සමීකරණ",     # Linear equations
        "සමීකරණ",           # Equations (general)
        "ගණිත",             # Mathematics (general)
        "බෙදීම",            # Division
        "ගුණ කිරීම",         # Multiplication
        "එකතු කිරීම",       # Addition
        "අඩු කිරීම",         # Subtraction
    ]

    # Math-related keywords (Sinhala)
    MATH_KEYWORDS = [
        # Arithmetic operations
        "විසඳන්න", "සොයන්න", "ගණනය", "පිළිතුර",

        # Math symbols and terms
        "සමීකරණය", "ගණිත", "අගය", "විසඳුම",

        # Variables and math notation
        "x", "y", "z", "a", "b", "c", "n",

        # Math operations
        "+", "-", "×", "÷", "=", "≠", "≤", "≥",
        "²", "³", "√", "∛",

        # Equation-related
        "log", "ln", "sin", "cos", "tan",

        # Numbers
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    ]

    # Non-math topics to reject
    BLOCKED_TOPICS = [
        # General knowledge
        "ඉතිහාසය", "history", "ජීව", "biology", "රසායන", "chemistry",
        "භෞතික", "physics", "විද්‍යාව", "science",

        # Social/personal
        "ආදරය", "love", "මිතුරු", "friends", "පවුල", "family",

        # Entertainment
        "සිනමා", "movie", "ගීත", "song", "ක්‍රීඩා", "sports",

        # Technology (non-math)
        "mobile", "phone", "computer", "game",

        # General chitchat
        "හෙලෝ", "hello", "කොහොමද", "how are you", "ස්තූතියි", "thanks",
        "හායි", "hi", "bye", "goodbye",

        # Visual/Diagram-based questions (cannot handle without images)
        "රූප", "diagram", "චිත්‍ර", "image", "පින්තූර", "picture",
        "රූප සටහන", "figure", "ප්‍රස්තාර", "graph", "ප්‍රස්තාරය", "chart",
        "හැඩ", "shape", "හැඩය", "shapes", "ත්‍රිකෝණ", "triangle",
        "චතුරස්‍ර", "square", "රවුම", "circle", "වෘත්ත", "rectangle",
        "ප්‍රතිරූප", "pattern", "ඒ පරිදි", "as shown", "දැක්වෙන",
        "ඉහත දැක්වෙන", "shown above", "පහත දැක්වෙන", "shown below",
        "රූපයේ", "in the figure", "චිත්‍රයේ", "in the diagram",
        "ජ්‍යාමිති", "geometry", "රේඛා ජ්‍යාමිති", "solid geometry",
        "කෝණ", "angle", "බිම", "area", "පරිමාව", "volume", "පරිමිතිය", "perimeter",
        "ත්‍රිකෝණමිති", "trigonometry", "sin", "cos", "tan", "sec", "cosec", "cot"
    ]

    def __init__(self, use_groq_validation: bool = True):
        """
        Initialize question filter

        Args:
            use_groq_validation: If True, uses Groq API for advanced filtering
        """
        self.use_groq_validation = use_groq_validation

        if use_groq_validation:
            try:
                from groq_validator import GroqValidator
                self.groq = GroqValidator()
                logger.info("✓ Groq-based question filtering enabled")
            except Exception as e:
                logger.warning(f"Groq not available for filtering: {e}")
                self.use_groq_validation = False
                self.groq = None
        else:
            self.groq = None

    def validate_question(self, question: str) -> Dict:
        """
        Validate if question is acceptable

        Returns:
            {
                'is_valid': bool,
                'reason': str (if not valid),
                'category': str ('math_topic', 'off_topic', 'chitchat', etc.)
            }
        """

        question_lower = question.lower()

        # 1. Check if question is empty or too short
        if len(question.strip()) < 3:
            return {
                'is_valid': False,
                'reason': 'කරුණාකර වලංගු ප්‍රශ්නයක් ඇතුළත් කරන්න.',
                'category': 'empty'
            }

        # 2. Check for blocked topics (immediate rejection)
        visual_keywords = ["රූප", "diagram", "චිත්‍ර", "image", "පින්තූර", "picture",
                          "රූප සටහන", "figure", "ප්‍රස්තාර", "graph", "හැඩ", "shape",
                          "ත්‍රිකෝණ", "triangle", "චතුරස්‍ර", "square", "වෘත්ත", "circle",
                          "රූපයේ", "චිත්‍රයේ", "දැක්වෙන", "shown", "ජ්‍යාමිති", "geometry",
                          "කෝණ", "angle", "බිම", "area", "පරිමාව", "volume", "ත්‍රිකෝණමිති", "trigonometry"]

        for blocked in self.BLOCKED_TOPICS:
            if blocked in question_lower:
                # Special message for visual/diagram questions
                if blocked in visual_keywords:
                    return {
                        'is_valid': False,
                        'reason': 'මට රූප, චිත්‍ර, හෝ ජ්‍යාමිතික හැඩ සහිත ප්‍රශ්න වලට පිළිතුරු දිය නොහැක. කරුණාකර ලඝුගණක, ශ්‍රීඝ්‍රතාවය, සමාන්තර ශ්‍රේණි හෝ සමීකරණ වැනි ගණිත ප්‍රශ්න අසන්න.',
                        'category': 'visual_question'
                    }

                return {
                    'is_valid': False,
                    'reason': f'මට හැකි වන්නේ ගණිත ප්‍රශ්න පමණක් පිළිතුරු දීමට පමණි. මම {", ".join(self.SUPPORTED_TOPICS[:3])} වැනි ගණිත විෂයන් ගැන උදව් කළ හැක.',
                    'category': 'off_topic'
                }

        # 3. Check if question contains math keywords
        has_math_keyword = any(keyword in question for keyword in self.MATH_KEYWORDS)

        if not has_math_keyword:
            # If no math keywords, might be chitchat or off-topic
            # Use Groq for advanced validation if available
            if self.use_groq_validation and self.groq:
                return self._groq_validate_question(question)
            else:
                return {
                    'is_valid': False,
                    'reason': 'මට ගණිත ප්‍රශ්න පමණක් පිළිතුරු දිය හැක. කරුණාකර ගණිත සම්බන්ධ ප්‍රශ්නයක් අසන්න.',
                    'category': 'no_math_keywords'
                }

        # 4. Check if question is about supported topics (optional strict mode)
        if config.STRICT_TOPIC_FILTERING:
            has_supported_topic = any(topic in question for topic in self.SUPPORTED_TOPICS)

            if not has_supported_topic:
                return {
                    'is_valid': False,
                    'reason': f'මට පහත ගණිත විෂයන් ගැන පමණක් උදව් කළ හැක:\n• {chr(10).join("• " + topic for topic in self.SUPPORTED_TOPICS[:5])}\n\nකරුණාකර මෙම විෂයන් සම්බන්ධ ප්‍රශ්නයක් අසන්න.',
                    'category': 'unsupported_topic'
                }

        # 5. All checks passed
        return {
            'is_valid': True,
            'reason': '',
            'category': 'math_question'
        }

    def _groq_validate_question(self, question: str) -> Dict:
        """
        Use Groq API to intelligently validate if question is math-related
        More accurate but slower
        """

        logger.info("Using Groq for advanced question validation...")

        try:
            validation_prompt = f"""පහත ප්‍රශ්නය ගණිත ප්‍රශ්නයක්ද නැද්ද තීරණය කරන්න:

**ප්‍රශ්නය:** {question}

**අනුමත ගණිත විෂයන්:**
- ලඝුගණක (Logarithms)
- ශ්‍රීඝ්‍රතාවය (Velocity/Speed)
- සමාන්තර ශ්‍රේණි (Arithmetic Progression)
- වර්ගජ සමීකරණ (Quadratic equations)
- රේඛීය සමීකරණ (Linear equations)
- මූලික ගණිතය (Basic arithmetic)

**නොමැති විෂයන්:** ඉතිහාසය, විද්‍යාව, තාක්ෂණය, සාමාන්‍ය කතාබස්

පහත JSON ආකෘතියෙන් පිළිතුරු දෙන්න:
```json
{{
  "is_math": true හෝ false,
  "is_supported_topic": true හෝ false,
  "category": "math", "science", "chitchat", "off_topic" වැනි එකක්,
  "explanation": "කෙටි පැහැදිලි කිරීමක්"
}}
```"""

            response = self.groq.client.chat.completions.create(
                model=self.groq.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a question classifier. Respond only in valid JSON format."
                    },
                    {
                        "role": "user",
                        "content": validation_prompt
                    }
                ],
                temperature=0.1,
                max_tokens=256
            )

            response_text = response.choices[0].message.content

            # Parse JSON response
            import json

            # Extract JSON from response
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            elif "{" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                json_text = response_text[json_start:json_end]
            else:
                json_text = response_text.strip()

            result = json.loads(json_text)

            # Determine validity
            is_math = result.get('is_math', False)
            is_supported = result.get('is_supported_topic', False)
            category = result.get('category', 'unknown')

            if not is_math:
                return {
                    'is_valid': False,
                    'reason': 'මට ගණිත ප්‍රශ්න පමණක් පිළිතුරු දිය හැක. කරුණාකර ගණිත සම්බන්ධ ප්‍රශ්නයක් අසන්න.',
                    'category': category
                }

            if config.STRICT_TOPIC_FILTERING and not is_supported:
                return {
                    'is_valid': False,
                    'reason': f'මට පහත ගණිත විෂයන් ගැන පමණක් උදව් කළ හැක:\n• ලඝුගණක\n• ශ්‍රීඝ්‍රතාවය\n• සමාන්තර ශ්‍රේණි\n• වර්ගජ සහ රේඛීය සමීකරණ\n\nකරුණාකර මෙම විෂයන් සම්බන්ධ ප්‍රශ්නයක් අසන්න.',
                    'category': 'unsupported_math_topic'
                }

            return {
                'is_valid': True,
                'reason': '',
                'category': 'math_question'
            }

        except Exception as e:
            logger.error(f"Groq validation failed: {e}")

            # Fallback to basic validation
            return {
                'is_valid': True,  # Allow question through if Groq fails
                'reason': '',
                'category': 'validation_error'
            }

    def get_rejection_message(self, category: str) -> str:
        """Get friendly rejection message based on category"""

        messages = {
            'empty': 'කරුණාකර වලංගු ප්‍රශ්නයක් ඇතුළත් කරන්න.',

            'off_topic': f'මට හැකි වන්නේ ගණිත ප්‍රශ්න පමණක් පිළිතුරු දීමට පමණි. \n\nමම උදව් කළ හැකි විෂයන්:\n• ලඝුගණක\n• ශ්‍රීඝ්‍රතාවය\n• සමාන්තර ශ්‍රේණි\n• වර්ගජ සහ රේඛීය සමීකරණ',

            'visual_question': 'මට රූප, චිත්‍ර, ජ්‍යාමිතික හැඩ හෝ ප්‍රස්තාර සහිත ප්‍රශ්න වලට පිළිතුරු දිය නොහැක.\n\nමම උදව් කළ හැකි විෂයන්:\n• ලඝුගණක\n• ශ්‍රීඝ්‍රතාවය\n• සමාන්තර ශ්‍රේණි\n• සමීකරණ (වර්ගජ සහ රේඛීය)\n\nකරුණාකර රූප අවශ්‍ය නොවන ගණිත ප්‍රශ්නයක් අසන්න.',

            'chitchat': 'ආයුබෝවන්! මම ගණිත ගුරුවරයෙකි. මට ගණිත ප්‍රශ්න වලට උදව් කළ හැක. කරුණාකර ගණිත සම්බන්ධ ප්‍රශ්නයක් අසන්න.',

            'no_math_keywords': 'මට ගණිත ප්‍රශ්න පමණක් පිළිතුරු දිය හැක. කරුණාකර සමීකරණ, ගණනය කිරීම් හෝ ගණිත ගැටළු සම්බන්ධ ප්‍රශ්නයක් අසන්න.',

            'unsupported_topic': f'මට පහත ගණිත විෂයන් ගැන පමණක් උදව් කළ හැක:\n• ලඝුගණක\n• ශ්‍රීඝ්‍රතාවය\n• සමාන්තර ශ්‍රේණි\n• වර්ගජ සහ රේඛීය සමීකරණ\n\nකරුණාකර මෙම විෂයන් සම්බන්ධ ප්‍රශ්නයක් අසන්න.',
        }

        return messages.get(category, 'කරුණාකර වලංගු ගණිත ප්‍රශ්නයක් අසන්න.')


# Quick test function
if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING QUESTION FILTER")
    print("="*70)

    filter = QuestionFilter(use_groq_validation=False)

    # Test cases
    test_questions = [
        ("x² + 5x + 6 = 0 විසඳන්න", True),  # Valid math
        ("log(x) = 2 විසඳන්න", True),  # Valid math (logarithms)
        ("ශ්‍රීඝ්‍රතාවය ගණනය කරන්න", True),  # Valid math (velocity)
        ("ඉතිහාසය ගැන කියන්න", False),  # Off-topic (history)
        ("හෙලෝ කොහොමද", False),  # Chitchat
        ("", False),  # Empty
        ("2 + 2", True),  # Simple math
    ]

    for question, expected_valid in test_questions:
        result = filter.validate_question(question)
        status = "✓" if result['is_valid'] == expected_valid else "✗"

        print(f"\n{status} Question: {question or '(empty)'}")
        print(f"  Valid: {result['is_valid']}")
        print(f"  Category: {result['category']}")
        if not result['is_valid']:
            print(f"  Reason: {result['reason']}")

    print("\n" + "="*70)
