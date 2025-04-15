import json
import jsonlines
import os
from pathlib import Path
from collections import defaultdict

def read_jsonl_file(file_path):
    """Read a JSONL file and return its contents as a list of dictionaries."""
    data = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data.append(obj)
    return data

def read_json_file(file_path):
    """Read a JSON file and return its contents."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def write_jsonl_file(data, output_path):
    """Write data to a JSONL file."""
    with jsonlines.open(output_path, mode='w') as writer:
        for item in data:
            writer.write(item)

def analyze_database_structure(data):
    """Analyze the structure of the combined database."""
    print("\nAnalyzing database structure...")
    
    # Track field frequencies
    field_frequencies = defaultdict(int)
    # Track unique values for certain fields
    unique_topics = set()
    unique_subtopics = set()
    
    for item in data:
        # Count field frequencies
        for field in item.keys():
            field_frequencies[field] += 1
            
        # Track unique topics and subtopics
        if 'failure_topic' in item:
            unique_topics.add(item['failure_topic'])
        if 'subtopics' in item:
            unique_subtopics.update(item['subtopics'])
    
    print("\nField Frequencies:")
    for field, count in field_frequencies.items():
        print(f"{field}: {count} records")
    
    print(f"\nUnique Topics: {len(unique_topics)}")
    print(f"Unique Subtopics: {len(unique_subtopics)}")
    
    # Print sample of unique topics and subtopics
    print("\nSample Topics:")
    for topic in list(unique_topics)[:5]:
        print(f"- {topic}")
    
    print("\nSample Subtopics:")
    for subtopic in list(unique_subtopics)[:5]:
        print(f"- {subtopic}")

def combine_learning_files():
    # List of files to combine
    files_to_combine = [
        'failure_insights_batch_001_fixed.jsonl',  # Using the fixed file
        'failure_insights_batch_002.jsonl',
        'category_examples_500.jsonl',
        'batch_001.jsonl',
        'topic_index_1M.jsonl',
        'massive_marketing_database.json',
        'marketing_ai_content_database_seed.json'
    ]
    
    combined_data = []
    total_records = 0
    
    # Create output directory if it doesn't exist
    output_dir = Path('combined_learning_data')
    output_dir.mkdir(exist_ok=True)
    
    # Process each file
    for file_name in files_to_combine:
        if not os.path.exists(file_name):
            print(f"Warning: {file_name} not found, skipping...")
            continue
            
        print(f"Processing {file_name}...")
        
        try:
            if file_name.endswith('.jsonl'):
                data = read_jsonl_file(file_name)
            else:
                data = read_json_file(file_name)
                # If the JSON file contains a single object, convert it to a list
                if isinstance(data, dict):
                    data = [data]
            
            combined_data.extend(data)
            total_records += len(data)
            print(f"Added {len(data)} records from {file_name}")
            
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")
    
    # Write combined data to a new JSONL file
    output_file = output_dir / 'combined_marketing_learning_database.jsonl'
    write_jsonl_file(combined_data, output_file)
    
    print(f"\nCombined {total_records} total records")
    print(f"Output written to: {output_file}")
    
    # Analyze the combined database structure
    analyze_database_structure(combined_data)

if __name__ == "__main__":
    combine_learning_files() 