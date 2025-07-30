import json
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
from utils import is_heading_numbering, is_title_case, is_all_caps

def create_training_data():
    """Create training data from debug files"""
    training_data = []
    
    # Look for debug files in output directory
    output_dir = "/app/output"
    if not os.path.exists(output_dir):
        print("No output directory found. Run PDF extraction first.")
        return None
    
    debug_files = [f for f in os.listdir(output_dir) if f.endswith('_debug.json')]
    
    if not debug_files:
        print("No debug files found. Run PDF extraction first.")
        return None
    
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
    
    return training_data

def train_model():
    """Train the ML model"""
    print("Creating training data from debug files...")
    training_data = create_training_data()
    
    if not training_data:
        print("No training data available. Using heuristics only.")
        return
    
    print(f"Found {len(training_data)} training samples")
    
    # Create DataFrame
    df = pd.DataFrame(training_data)
    
    # Prepare features
    features = ['size', 'is_bold', 'is_all_caps', 'is_title_case', 'is_numbered', 'x0', 'y0']
    X = df[features]
    y = df['label']
    
    # Train model
    print("Training RandomForest model...")
    clf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
    clf.fit(X, y)
    
    # Save model
    os.makedirs('/app/models', exist_ok=True)
    joblib.dump(clf, '/app/models/heading_classifier.joblib')
    print("✓ Model saved to: /app/models/heading_classifier.joblib")
    
    # Test accuracy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf_test = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
    clf_test.fit(X_train, y_train)
    accuracy = clf_test.score(X_test, y_test)
    print(f"✓ Model accuracy: {accuracy:.2f}")
    
    # Show feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature importance:")
    for _, row in feature_importance.iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")

if __name__ == '__main__':
    train_model() 