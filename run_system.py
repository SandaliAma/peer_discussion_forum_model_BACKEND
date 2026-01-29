"""
Main Runner Script - Complete System Control
"""

import sys
import os

def print_menu():
    """Print main menu"""
    print("\n" + "="*70)
    print("SINHALM MATHEMATICS RAG SYSTEM")
    print("="*70)
    print("\nChoose an option:")
    print("  1. Quick Test (Test the system)")
    print("  2. Fine-Tune Model (Train on your data)")
    print("  3. Start API Server")
    print("  4. Rebuild RAG Index")
    print("  5. System Info")
    print("  6. Compare Models (Original vs Fine-tuned)")
    print("  7. Exit")
    print("="*70)

def quick_test():
    """Run quick test"""
    print("\nStarting Quick Test...\n")
    os.system("python quick_test.py")

def finetune():
    """Run fine-tuning"""
    print("\nStarting Fine-Tuning Process...\n")
    print("⚠️  WARNING: This will take time on CPU!")
    print("   Estimated: 1-4 hours depending on data size")
    print("   You can stop anytime with Ctrl+C\n")
    
    confirm = input("Continue with fine-tuning? (y/n): ")
    if confirm.lower() == 'y':
        os.system("python fine_tuner.py")
    else:
        print("Fine-tuning cancelled.")

def start_api():
    """Start API server"""
    print("\nStarting API Server...\n")
    os.system("python api_server.py")

def rebuild_index():
    """Rebuild RAG index"""
    print("\nRebuilding RAG Index...\n")
    
    from math_rag_system import SystemBuilder
    
    print("This will rebuild the search index.")
    confirm = input("Continue? (y/n): ")
    
    if confirm.lower() == 'y':
        SystemBuilder.build(rebuild_index=True)
        print("\n✓ Index rebuilt successfully!")
    else:
        print("Cancelled.")

def system_info():
    """Display system information"""
    import config
    from math_rag_system import MathProblemProcessor
    
    print("\n" + "="*70)
    print("SYSTEM INFORMATION")
    print("="*70)
    
    # Configuration
    print("\n📁 Configuration:")
    print(f"  Project: {config.BASE_DIR}")
    print(f"  Data file: {config.MATH_PROBLEMS_JSON}")
    print(f"  SinhaLM model: {config.SINHALM_MODEL_PATH}")
    print(f"  Fine-tuned model: {config.FINETUNED_MODEL_PATH}")
    print(f"  RAG index: {config.RAG_INDEX_DIR}")
    print(f"  HF Cache: {os.environ.get('HF_HOME', 'Not set')}")
    
    # File status
    print("\n✅ File Status:")
    print(f"  Data file: {'✓ Found' if os.path.exists(config.MATH_PROBLEMS_JSON) else '❌ Not found'}")
    print(f"  SinhaLM model: {'✓ Found' if os.path.exists(config.SINHALM_MODEL_PATH) else '❌ Not found'}")
    print(f"  Fine-tuned model: {'✓ Found' if os.path.exists(config.FINETUNED_MODEL_PATH) else '❌ Not trained yet'}")
    print(f"  RAG index: {'✓ Built' if os.path.exists(os.path.join(config.RAG_INDEX_DIR, 'faiss.index')) else '❌ Not built'}")
    
    # Disk space
    import shutil
    total_c, used_c, free_c = shutil.disk_usage("C:\\")
    total_d, used_d, free_d = shutil.disk_usage("D:\\")
    
    print("\n💾 Disk Space:")
    print(f"  C: drive: {free_c // (2**30)} GB free")
    print(f"  D: drive: {free_d // (2**30)} GB free")
    
    # Problem statistics
    if os.path.exists(config.MATH_PROBLEMS_JSON):
        try:
            processor = MathProblemProcessor()
            problems = processor.process_all(config.MATH_PROBLEMS_JSON)
            stats = processor.get_statistics(problems)
            
            print("\n📊 Problem Statistics:")
            print(f"  Total problems: {stats['total']}")
            print(f"  Topics: {len(stats['topics'])}")
            
            for topic, count in sorted(stats['topics'].items(), key=lambda x: x[1], reverse=True):
                print(f"    - {topic}: {count}")
                
            if stats['sub_topics']:
                print(f"  Sub-topics: {len(stats['sub_topics'])}")
                
        except Exception as e:
            print(f"\n❌ Error reading problems: {e}")
    
    print("\n" + "="*70)

def compare_models():
    """Compare original and fine-tuned models"""
    import config
    
    if not os.path.exists(config.FINETUNED_MODEL_PATH):
        print("\n❌ Fine-tuned model not found!")
        print("   Please fine-tune the model first (Option 2)")
        return
    
    print("\nStarting model comparison...")
    print("You'll be able to test the same questions with both models.\n")
    
    os.system("python quick_test.py")

def main():
    """Main function"""
    
    # Check configuration first
    import config
    if not config.validate_config():
        print("\n❌ Configuration errors detected!")
        print("Please fix the issues before proceeding.")
        return
    
    while True:
        print_menu()
        
        try:
            choice = input("\nEnter choice (1-7): ").strip()
            
            if choice == '1':
                quick_test()
            elif choice == '2':
                finetune()
            elif choice == '3':
                start_api()
            elif choice == '4':
                rebuild_index()
            elif choice == '5':
                system_info()
            elif choice == '6':
                compare_models()
            elif choice == '7':
                print("\nExiting... අවසන්!")
                break
            else:
                print("\n❌ Invalid choice. Please enter 1-7.")
                
        except KeyboardInterrupt:
            print("\n\nExiting... අවසන්!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()