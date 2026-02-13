"""
Simple test script for question filtering
Run this to see the filter in action
"""

from question_filter import QuestionFilter
import config

def test_filtering():
    """Test question filtering with various questions"""

    print("\n" + "="*70)
    print("QUESTION FILTER TEST")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Filtering enabled: {config.QUESTION_FILTERING_ENABLED}")
    print(f"  Strict mode: {config.STRICT_TOPIC_FILTERING}")
    print(f"  Groq filtering: {config.USE_GROQ_FOR_FILTERING}")
    print("="*70)

    # Initialize filter (without Groq for fast testing)
    filter = QuestionFilter(use_groq_validation=False)

    # Test cases: (question, should_be_valid, description)
    test_cases = [
        # Valid math questions
        ("x² + 5x + 6 = 0 විසඳන්න", True, "Quadratic equation"),
        ("log(x) = 2 නම් x හි අගය සොයන්න", True, "Logarithm question"),
        ("ශ්‍රීඝ්‍රතාවය ගණනය කරන්න", True, "Velocity calculation"),
        ("2x + 3 = 7 විසඳන්න", True, "Linear equation"),
        ("2 + 2 = කීයද?", True, "Simple arithmetic"),
        ("(3/a) + (2/a) = 1/2 විසඳන්න", True, "Fraction equation"),

        # Off-topic questions (should be filtered)
        ("ඉතිහාසය ගැන කියන්න", False, "History question"),
        ("ජීව විද්‍යාව ගැන උගන්වන්න", False, "Biology question"),
        ("හෙලෝ කොහොමද", False, "Chitchat greeting"),
        ("ආදරය කියන්නේ මොකක්ද", False, "Personal question"),
        ("What is the capital of Sri Lanka?", False, "General knowledge"),
        ("සිනමා පැත්තේ මොනවා තියෙනවද", False, "Entertainment"),

        # Visual/Diagram questions (should be filtered - cannot handle without images)
        ("රූපයේ ත්‍රිකෝණයේ කෝණය සොයන්න", False, "Triangle in diagram"),
        ("දී ඇති චිත්‍රයේ පරිමාව ගණනය කරන්න", False, "Area from diagram"),
        ("ප්‍රස්තාරය බලා පිළිතුර දෙන්න", False, "Graph-based question"),
        ("ඉහත දැක්වෙන හැඩයේ පරිමිතිය", False, "Shape perimeter"),
        ("රූප සටහනේ අගයන් භාවිතා කරන්න", False, "Using figure values"),
        ("ජ්‍යාමිතික හැඩ ගැන", False, "Geometry shapes"),

        # Edge cases
        ("", False, "Empty question"),
        ("x", False, "Too short"),
        ("help", False, "General help"),
    ]

    print("\n" + "="*70)
    print("TEST RESULTS")
    print("="*70)

    passed = 0
    failed = 0

    for question, expected_valid, description in test_cases:
        result = filter.validate_question(question)

        is_correct = (result['is_valid'] == expected_valid)
        status = "✓ PASS" if is_correct else "✗ FAIL"

        if is_correct:
            passed += 1
        else:
            failed += 1

        print(f"\n{status} | {description}")
        print(f"  Question: {question or '(empty)'}")
        print(f"  Expected: {'Valid' if expected_valid else 'Filtered'}")
        print(f"  Got: {'Valid' if result['is_valid'] else 'Filtered'}")
        print(f"  Category: {result['category']}")

        if not result['is_valid']:
            print(f"  Reason: {result['reason'][:80]}...")

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"  Total tests: {len(test_cases)}")
    print(f"  ✓ Passed: {passed}")
    print(f"  ✗ Failed: {failed}")
    print(f"  Success rate: {passed/len(test_cases)*100:.1f}%")
    print("="*70 + "\n")


def interactive_test():
    """Interactive testing mode"""

    print("\n" + "="*70)
    print("INTERACTIVE FILTER TEST")
    print("="*70)
    print("Enter questions to test filtering (or 'quit' to exit)\n")

    # Initialize with Groq for more accurate testing
    print("Initializing filter...")
    filter = QuestionFilter(use_groq_validation=config.USE_GROQ_FOR_FILTERING)
    print("✓ Ready!\n")

    while True:
        question = input("\nප්‍රශ්නය: ").strip()

        if question.lower() in ['quit', 'exit', 'q']:
            print("අවසන්!")
            break

        if not question:
            continue

        print("\nValidating...")
        result = filter.validate_question(question)

        print("\n" + "-"*70)
        if result['is_valid']:
            print("✓ VALID - Question will be processed")
            print(f"  Category: {result['category']}")
        else:
            print("✗ FILTERED - Question will be rejected")
            print(f"  Category: {result['category']}")
            print(f"\nRejection message:")
            print(f"  {result['reason']}")
        print("-"*70)


if __name__ == "__main__":
    import sys

    print("\n" + "="*70)
    print("QUESTION FILTER TESTING TOOL")
    print("="*70)
    print("\nOptions:")
    print("  1. Run automated tests")
    print("  2. Interactive testing mode")
    print("  3. Both")
    print("="*70)

    try:
        choice = input("\nEnter choice (1-3): ").strip()

        if choice == '1':
            test_filtering()
        elif choice == '2':
            interactive_test()
        elif choice == '3':
            test_filtering()
            input("\nPress Enter to start interactive mode...")
            interactive_test()
        else:
            print("Invalid choice, running automated tests...")
            test_filtering()

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
