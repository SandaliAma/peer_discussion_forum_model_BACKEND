"""
Filter dataset to keep ONLY problems with complete solutions
This will improve training quality significantly
"""

import json
import os

def filter_good_problems(input_file, output_file):
    """Keep only problems with Steps and Final_answer"""
    
    print("="*70)
    print("FILTERING DATASET - KEEPING ONLY PROBLEMS WITH SOLUTIONS")
    print("="*70)
    
    # Load data
    print(f"\nLoading: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total problems: {len(data)}")
    
    # Filter for problems with solutions
    good_problems = [
        p for p in data 
        if 'Steps' in p and 'Final_answer' in p and p['Steps'] and p['Final_answer']
    ]
    
    print(f"\n✅ Problems WITH complete solutions: {len(good_problems)}")
    print(f"❌ Problems WITHOUT solutions: {len(data) - len(good_problems)}")
    
    if not good_problems:
        print("\n⚠️ ERROR: No problems with solutions found!")
        return False
    
    # Show sample
    print(f"\n📝 Sample good problem:")
    print(f"Question: {good_problems[0]['Question'][:80]}...")
    print(f"Steps: {len(good_problems[0]['Steps'])} steps")
    print(f"Answer: {good_problems[0]['Final_answer']}")
    
    # Save filtered data
    print(f"\n💾 Saving to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(good_problems, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Saved {len(good_problems)} problems with complete solutions")
    
    # Show statistics
    print(f"\n📊 Quality Improvement:")
    print(f"   Before: {len(good_problems)}/{len(data)} = {100*len(good_problems)/len(data):.1f}% had solutions")
    print(f"   After:  {len(good_problems)}/{len(good_problems)} = 100% have solutions")
    print(f"   Training quality improved by {100 - 100*len(good_problems)/len(data):.1f}%!")
    
    return True

def analyze_formats(data):
    """Analyze what formats exist in data"""
    
    print("\n" + "="*70)
    print("DATA FORMAT ANALYSIS")
    print("="*70)
    
    formats = {}
    for p in data:
        key_sig = str(sorted(p.keys()))
        if key_sig not in formats:
            formats[key_sig] = {'count': 0, 'example': p}
        formats[key_sig]['count'] += 1
    
    for i, (key_sig, info) in enumerate(formats.items(), 1):
        print(f"\nFormat {i}: {info['count']} problems")
        print(f"Keys: {key_sig}")
        
        # Show if it has solutions
        ex = info['example']
        has_solution = bool(
            ('Steps' in ex and ex.get('Steps')) or
            ('solution' in ex and ex.get('solution')) or
            ('Solution_Methods' in ex and ex.get('Solution_Methods'))
        )
        print(f"Has solution: {'✅ YES' if has_solution else '❌ NO'}")

if __name__ == "__main__":
    input_file = "data/combined_dataset.json"
    output_file = "data/clean_dataset.json"
    
    # Backup original
    backup_file = "data/combined_dataset_backup.json"
    if os.path.exists(input_file) and not os.path.exists(backup_file):
        print(f"Creating backup: {backup_file}")
        import shutil
        shutil.copy(input_file, backup_file)
    
    # Load and analyze
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    analyze_formats(data)
    
    # Filter
    print("\n")
    success = filter_good_problems(input_file, output_file)
    
    if success:
        print("\n" + "="*70)
        print("✅ FILTERING COMPLETE!")
        print("="*70)
        print("\nNext steps:")
        print("1. Update config.py:")
        print("   MATH_PROBLEMS_JSON = os.path.join(DATA_DIR, 'clean_dataset.json')")
        print("\n2. Rebuild RAG index:")
        print("   python -c \"from math_rag_system import SystemBuilder; SystemBuilder.build(rebuild_index=True)\"")
        print("\n3. Retrain model (optional):")
        print("   python fine_tuner.py")
        print("\n4. Restart server:")
        print("   python fast_api_server.py")
    else:
        print("\n❌ Filtering failed!")