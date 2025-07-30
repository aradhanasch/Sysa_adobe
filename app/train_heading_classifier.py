import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models/heading_classifier.joblib')

FEATURES = [
    'size', 'is_bold', 'is_all_caps', 'is_title_case', 'is_numbered',
    'x0', 'y0', 'text_len'
]

LABELS = ['H1', 'H2', 'H3', 'OTHER']

def extract_features(df):
    df['is_bold'] = df['is_bold'].astype(int)
    df['is_all_caps'] = df['is_all_caps'].astype(int)
    df['is_title_case'] = df['is_title_case'].astype(int)
    df['is_numbered'] = df['is_numbered'].astype(int)
    df['text_len'] = df['text'].apply(len)
    return df[FEATURES]

def main():
    # User must provide labeled CSV: text, size, is_bold, is_all_caps, is_title_case, is_numbered, x0, y0, label
    data_path = input('Enter path to labeled heading CSV: ').strip()
    df = pd.read_csv(data_path)
    X = extract_features(df)
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, labels=LABELS))
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f'Model saved to {MODEL_PATH}')

if __name__ == '__main__':
    main() 