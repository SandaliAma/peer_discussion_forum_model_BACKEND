"""
Merge all JSON files in data folder into one unified dataset
Handles multiple JSON formats automatically
FIXED VERSION: Handles string sub_questions
"""

import json
import os
from pathlib import Path

# Configuration
DATA_DIR = "data"
OUTPUT_FILE = "data/combined_dataset.json"

def load_json_file(filepath):
    """Load a single JSON file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"   ✓ Loaded: {filepath.name}")
        return data
    except Exception as e:
        print(f"   ❌ Error loading {filepath.name}: {e}")
        return None

def extract_problems(data, filename):
    """Extract problems from various JSON structures - FIXED VERSION"""
    problems = []
    
    if data is None:
        return problems
    
    # Handle different JSON structures
    if isinstance(data, list):
        # Direct list of problems
        problems = data
    elif isinstance(data, dict):
        # Check for various wrapper keys
        if 'examples' in data:
            problems = data['examples']
        elif 'exercises' in data:
            # Special handling for exercises format
            exercises = data['exercises']
            if isinstance(exercises, list):
                # Flatten sub_questions into individual problems
                for exercise in exercises:
                    metadata = exercise.get('metadata', {})
                    sub_questions = metadata.get('sub_questions', [])
                    
                    if sub_questions:
                        # Create individual problems from sub-questions
                        for i, sq in enumerate(sub_questions, 1):
                            # FIX: Handle both dict and string sub_questions
                            if isinstance(sq, dict):
                                problems.append({
                                    'question': sq.get('question', ''),
                                    'topic': exercise.get('topic', 'ප්‍රතිශත'),
                                    'sub_topic': exercise.get('type', ''),
                                    'main_context': metadata.get('main_question', ''),
                                    'source_file': filename,
                                    'source_type': 'extracted_exercise'
                                })
                            elif isinstance(sq, str):
                                # If sub_question is just a string
                                problems.append({
                                    'question': sq,
                                    'topic': exercise.get('topic', 'ප්‍රතිශත'),
                                    'sub_topic': exercise.get('type', ''),
                                    'main_context': metadata.get('main_question', ''),
                                    'source_file': filename,
                                    'source_type': 'extracted_exercise_string'
                                })
                    else:
                        # Use the exercise as-is
                        problems.append(exercise)
            else:
                problems = [exercises]
        elif 'problems' in data:
            problems = data['problems']
        elif 'data' in data:
            problems = data['data']
        else:
            # Single problem object
            problems = [data]
    
    # Add source file info to each problem
    for problem in problems:
        if isinstance(problem, dict) and 'source_file' not in problem:
            problem['source_file'] = filename
    
    return problems

def detect_format(problem):
    """Detect which format a problem uses"""
    if not isinstance(problem, dict):
        return 'non_dict'
    elif 'Question_ID' in problem or 'Question_Text' in problem:
        return 'new_format'
    elif 'Question' in problem:
        return 'capital_format'
    elif 'question' in problem:
        return 'old_format'
    else:
        return 'unknown'

def clean_problems(problems):
    """Clean and filter problems - remove non-dict entries"""
    cleaned = []
    
    for problem in problems:
        if isinstance(problem, dict):
            cleaned.append(problem)
        else:
            print(f"   ⚠️  Skipping non-dict problem: {type(problem)} - {str(problem)[:50]}...")
    
    return cleaned

def merge_all_datasets():
    """Merge all JSON files in data directory"""
    
    print("\n" + "="*70)
    print("MERGING ALL DATASET FILES")
    print("="*70)
    
    data_path = Path(DATA_DIR)
    
    if not data_path.exists():
        print(f"\n❌ Error: Directory '{DATA_DIR}' not found!")
        print(f"   Please create the directory or check the path.")
        return None
    
    # Get all JSON files
    json_files = list(data_path.glob("*.json"))
    
    # Exclude output file if it already exists
    json_files = [f for f in json_files if f.name != os.path.basename(OUTPUT_FILE)]
    
    if not json_files:
        print(f"\n❌ No JSON files found in '{DATA_DIR}' directory!")
        return None
    
    print(f"\n📁 Found {len(json_files)} JSON files:")
    for f in json_files:
        print(f"   - {f.name}")
    
    all_problems = []
    stats = {}
    format_stats = {}
    
    print("\n📊 Processing files...")
    print("-"*70)
    
    for json_file in json_files:
        filename = json_file.name
        print(f"\n📄 {filename}")
        
        # Load file
        data = load_json_file(json_file)
        
        if data is None:
            continue
        
        # Extract problems
        problems = extract_problems(data, filename)
        
        # Clean problems (remove non-dict entries)
        problems = clean_problems(problems)
        
        if not problems:
            print(f"   ⚠️  No valid problems found")
            continue
        
        # Detect formats
        formats_in_file = {}
        for problem in problems:
            fmt = detect_format(problem)
            formats_in_file[fmt] = formats_in_file.get(fmt, 0) + 1
        
        print(f"   ✓ {len(problems)} valid problems")
        print(f"   Formats: {formats_in_file}")
        
        # Add to collection
        all_problems.extend(problems)
        stats[filename] = len(problems)
        
        # Update format stats
        for fmt, count in formats_in_file.items():
            format_stats[fmt] = format_stats.get(fmt, 0) + count
    
    if not all_problems:
        print("\n❌ No valid problems found in any file!")
        return None
    
    print("\n" + "="*70)
    print(f"✅ MERGE COMPLETE: {len(all_problems)} total problems")
    print("="*70)
    
    # Show statistics
    print("\n📊 Problems per file:")
    for filename, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
        print(f"   {filename:45s} : {count:4d} problems")
    
    print("\n📋 Format distribution:")
    for fmt, count in sorted(format_stats.items()):
        print(f"   {fmt:20s} : {count:4d} problems")
    
    # Topic statistics (if available)
    topic_counts = {}
    sub_topic_counts = {}
    
    for p in all_problems:
        # Try different topic field names
        topic = p.get('Lesson_Topic') or p.get('topic') or p.get('Topic') or 'Unknown'
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        # Count sub-topics
        sub_topic = p.get('sub_topic') or p.get('Sub_Topic') or ''
        if sub_topic:
            sub_topic_counts[sub_topic] = sub_topic_counts.get(sub_topic, 0) + 1
    
    if len(topic_counts) > 1 or 'Unknown' not in topic_counts:
        print("\n📚 Topics found (top 15):")
        for topic, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:15]:
            print(f"   {topic:45s} : {count:4d} problems")
        
        if len(topic_counts) > 15:
            print(f"   ... and {len(topic_counts) - 15} more topics")
    
    if sub_topic_counts:
        print("\n📚 Sub-Topics found (top 10):")
        for sub_topic, count in sorted(sub_topic_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"   {sub_topic:45s} : {count:4d} problems")
        
        if len(sub_topic_counts) > 10:
            print(f"   ... and {len(sub_topic_counts) - 10} more sub-topics")
    
    # Check if problems have required fields
    required_fields = ['question', 'solution', 'final_answer']
    missing_fields = {field: 0 for field in required_fields}
    
    for p in all_problems:
        for field in required_fields:
            if not p.get(field):
                missing_fields[field] += 1
    
    print("\n📊 Required fields check:")
    for field, missing in missing_fields.items():
        has_count = len(all_problems) - missing
        print(f"   {field:15s}: {has_count}/{len(all_problems)} problems have it")
    
    # Save merged dataset
    print(f"\n💾 Saving to: {OUTPUT_FILE}")
    
    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_problems, f, ensure_ascii=False, indent=2)
        
        print("✅ Saved successfully!")
        
        # Show file size
        file_size = output_path.stat().st_size
        print(f"📦 File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
        
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    return all_problems

def main():
    """Main function"""
    
    try:
        merged = merge_all_datasets()
        
        if merged:
            print("\n" + "="*70)
            print("✅ ALL DONE!")
            print("="*70)
            print(f"\n📌 Next steps:")
            print(f"   1. The merged dataset has {len(merged)} problems")
            print(f"   2. Check config.py to ensure it points to: {OUTPUT_FILE}")
            print(f"   3. Run: python fine_tuner.py to train the model")
            print()
        else:
            print("\n❌ Merge failed. Please check the errors above.")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()