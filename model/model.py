import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

print("Model script started.")

def load_data():
    """
    Loads and prepares the synthetic credit card transaction data.
    """
    print("Loading synthetic data...")
    np.random.seed(42)
    num_transactions = 10000
    # This generates a dataset with the same structure as the original Kaggle dataset.
    data = pd.DataFrame({
        'Time': np.arange(num_transactions),
        **{f'V{i}': np.random.normal(0, 1, num_transactions) for i in range(1, 29)},
        'Amount': np.random.uniform(1, 2000, num_transactions),
        'Class': 0
    })

    # Introduce fraudulent transactions
    num_fraud = 50
    fraud_indices = np.random.choice(data.index, num_fraud, replace=False)
    data.loc[fraud_indices, 'Class'] = 1
    data.loc[fraud_indices, 'Amount'] *= np.random.uniform(2, 5, num_fraud)
    for col in [f'V{i}' for i in range(1, 29)]:
        data.loc[fraud_indices, col] *= np.random.uniform(1.5, 3)

    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    print("Data loading complete.")
    return data

def train_and_save_model(data):
    """
    Splits data, trains a RandomForestClassifier, and saves the model and test data.
    """
    print("Preparing data for training...")
    X = data.drop(columns=['Time', 'Class'])
    y = data['Class']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print("Training the Random Forest model...")
    # class_weight='balanced' is crucial for handling imbalanced data
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
    model.fit(X_train, y_train)
    print("Model training complete.")

    # Define the path to save the model artifact
    # This saves it in the main project directory
    model_path = os.path.join(os.path.dirname(__file__), '..', 'fraud_model.pkl')

    # Save the trained model and the test set for the app to use
    print(f"Saving model and test data to {model_path}...")
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'X_test': X_test, 'y_test': y_test}, f)
    print("Model artifact saved successfully.")


# Main execution block
if __name__ == '__main__':
    # This code runs only when you execute `python model/model.py` directly
    dataset = load_data()
    train_and_save_model(dataset)
    print("Model script finished.")
