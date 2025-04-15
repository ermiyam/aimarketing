import json
import jsonlines
from pathlib import Path

def fix_jsonl_file(input_file, output_file):
    """Fix JSONL format issues in the input file and write to output file."""
    fixed_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        current_line = ''
        for line in f:
            # Remove any newlines and whitespace
            line = line.strip()
            if not line:
                continue
                
            current_line += line
            
            try:
                # Try to parse the current line as JSON
                data = json.loads(current_line)
                fixed_data.append(data)
                current_line = ''
            except json.JSONDecodeError:
                # If parsing fails, continue accumulating the line
                continue
    
    # Write the fixed data to a new file
    with jsonlines.open(output_file, mode='w') as writer:
        for item in fixed_data:
            writer.write(item)
    
    return len(fixed_data)

if __name__ == "__main__":
    input_file = 'failure_insights_batch_001.jsonl'
    output_file = 'failure_insights_batch_001_fixed.jsonl'
    
    print(f"Fixing {input_file}...")
    num_records = fix_jsonl_file(input_file, output_file)
    print(f"Fixed {num_records} records and saved to {output_file}") 