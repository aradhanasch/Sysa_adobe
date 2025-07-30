import json
import os
import pandas as pd
from utils import is_heading_numbering, is_title_case, is_all_caps

def create_training_data_from_debug():
    """Create training data from existing debug files"""
    training_data = []
    
    # Look for debug files in output directory
    output_dir = "app/output"
    debug_files = [f for f in os.listdir(output_dir) if f.endswith('_debug.json')]
    
    for debug_file in debug_files:
        with open(os.path.join(output_dir, debug_file), 'r', encoding='utf-8') as f:
            debug_data = json.load(f)
        
        for item in debug_data:
            if 'assigned_level' in item and item['assigned_level']:
                # Extract features
                features = {
                    'text': item['text'],
                    'size': item['size'],
                    'is_bold': int(bool(item['flags'] & 2)),
                    'is_all_caps': int(is_all_caps(item['text'])),
                    'is_title_case': int(is_title_case(item['text'])),
                    'is_numbered': int(is_heading_numbering(item['text'])),
                    'x0': item['x0'],
                    'y0': item['y0'],
                    'label': item['assigned_level']
                }
                training_data.append(features)
    
    # Create DataFrame and save
    df = pd.DataFrame(training_data)
    df.to_csv('training_data.csv', index=False)
    print(f"Created training data with {len(training_data)} samples")
    print("Saved to: training_data.csv")
    
    # Show sample
    print("\nSample data:")
    print(df.head())
    
    return df

if __name__ == '__main__':
    create_training_data_from_debug() 