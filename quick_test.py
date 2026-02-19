"""
Quick Test Script - Test both original and fine-tuned models
"""

import os
from math_rag_system import SystemBuilder
import config

def print_result(question: str, result: dict):
    """Pretty print a result"""
    print("\n" + "="*70)
    print(f"ප්‍රශ්නය: {question}")
    print("="*70)
    
    if result['status'] == 'success':
        print(f"\n Model: {result['model_used']}")
        print(f"\n පිළිතුර:\n")
        print(result['answer'])
        
        print(f"\n සමාන ගැටළු: {result['num_retrieved']}")
        if result['similar_problems']:
            print("\nසමාන ගැටළු:")
            for i, sim in enumerate(result['similar_problems'][:3], 1):
                problem = sim['problem']
                print(f"  {i}. {problem['question'][:60]}...")
                print(f"     සමානත්වය: {sim['similarity']:.2%}")
    else:
        print(f"\n දෝෂය: {result.get('answer', 'Unknown error')}")
    
    print("="*70 + "\n")

def main():
    """Run quick tests"""
    
    print("\n" + "="*70)
    print("QUICK TEST - SINHALM MATH RAG SYSTEM")
    print("="*70 + "\n")
    
    # Check if fine-tuned model exists
    finetuned_exists = os.path.exists(config.FINETUNED_MODEL_PATH)
    
    if finetuned_exists:
        print("✓ Fine-tuned model detected!")
        print("\nWhich model would you like to test?")
        print("  1. Original SinhaLM")
        print("  2. Fine-tuned model")
        print("  3. Both (compare)")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == '1':
            use_finetuned = False
            test_both = False
        elif choice == '2':
            use_finetuned = True
            test_both = False
        elif choice == '3':
            test_both = True
            use_finetuned = False  # Start with original
        else:
            print("Invalid choice, using original model")
            use_finetuned = False
            test_both = False
    else:
        print("No fine-tuned model found. Using original SinhaLM.")
        use_finetuned = False
        test_both = False
    
    # Build system
    print("\nBuilding system (first time takes 2-3 minutes)...\n")
    rag_system = SystemBuilder.build(use_finetuned=use_finetuned)
    
    # Test questions
    test_questions = [
        "x² + 5x + 6 = 0 විසඳන්න",
        "(1/2)m + (2/3)n = 1 සමීකරණය විසඳන්න",
        "2x² + 3x - 9 = 0 වර්ගජ සමීකරණය සාධක භාවිතයෙන් විසඳන්න"
    ]
    
    print("\n" + "="*70)
    print("TESTING WITH SAMPLE QUESTIONS")
    print("="*70)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\nTest {i}/{len(test_questions)}:")
        
        if test_both:
            # Test with original
            print("\n--- ORIGINAL SINHALM ---")
            result = rag_system.answer_question(question)
            print_result(question, result)
            
            # Rebuild with fine-tuned
            print("Switching to fine-tuned model...")
            rag_system = SystemBuilder.build(use_finetuned=True)
            
            print("\n--- FINE-TUNED MODEL ---")
            result = rag_system.answer_question(question)
            print_result(question, result)
            
            # Switch back for next question
            rag_system = SystemBuilder.build(use_finetuned=False)
        else:
            result = rag_system.answer_question(question)
            print_result(question, result)
        
        if i < len(test_questions):
            input("Press Enter to continue to next question...")
    
    # Interactive mode
    print("\n" + "="*70)
    print("INTERACTIVE MODE")
    print("="*70)
    print("Enter your questions (or 'quit' to exit)\n")
    
    while True:
        question = input("\nප්‍රශ්නය: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("අවසන්!")
            break
        
        if not question:
            continue
        
        result = rag_system.answer_question(question)
        print_result(question, result)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()