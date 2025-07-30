import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

def quick_train():
    """Quick training using auto-generated data"""
    
    # Create training data from debug files
    from create_training_data import create_training_data_from_debug
    df = create_training_data_from_debug()
    
    if len(df) < 10:
        print("Not enough training data. Using heuristics only.")
        return
    
    # Prepare features
    features = ['size', 'is_bold', 'is_all_caps', 'is_title_case', 'is_numbered', 'x0', 'y0']
    X = df[features]
    y = df['label']
    
    # Train model
    print("Training RandomForest model...")
    clf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
    clf.fit(X, y)
    
    # Save model
    os.makedirs('app/models', exist_ok=True)
    joblib.dump(clf, 'app/models/heading_classifier.joblib')
    print("✓ Model saved to: app/models/heading_classifier.joblib")
    
    # Test accuracy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf_test = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
    clf_test.fit(X_train, y_train)
    accuracy = clf_test.score(X_test, y_test)
    print(f"✓ Model accuracy: {accuracy:.2f}")

if __name__ == '__main__':
    quick_train() 