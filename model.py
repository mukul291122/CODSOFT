import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import joblib

# Define file paths
train_data_path = r'C:\Mukul\Movie genre\dataset\train_data.csv'
test_data_path = r'C:\Mukul\Movie genre\dataset\test_data.csv'
test_solution_path = r'C:\Mukul\Movie genre\dataset\test_data_solution.csv'
model_path = r'logistic_regression_model.pkl'
vectorizer_path = r'tfidf_vectorizer.pkl'
label_encoder_path = r'label_encoder.pkl'

def load_data(file_path):
    """Load and parse data from a CSV file."""
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    """Combine TITLE and DESCRIPTION into a single feature."""
    if 'DESCRIPTION' in data.columns:
        data['text'] = data['TITLE'] + ' ' + data['DESCRIPTION']
    else:
        data['text'] = data['TITLE']  # For test data or any other scenario
    return data

def main():
    # Load and preprocess training data
    train_data = load_data(train_data_path)
    train_data = preprocess_data(train_data)
    
    X_train = train_data['text']
    y_train = train_data['GENRE']
    
    # Vectorize text data
    vectorizer = TfidfVectorizer(max_features=10000, min_df=5, max_df=0.7, ngram_range=(1, 2))
    X_train_vectorized = vectorizer.fit_transform(X_train)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    
    # Train Logistic Regression model
    model = LogisticRegression(max_iter=100, multi_class='auto', solver='liblinear')
    model.fit(X_train_vectorized, y_train_encoded)
    
    # Save the model and vectorizer
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(label_encoder, label_encoder_path)
    
    # Load and preprocess test data
    test_data = load_data(test_data_path)
    test_data = preprocess_data(test_data)
    
    # Load and preprocess test solution data
    test_solution_data = load_data(test_solution_path)
    test_solution_data = preprocess_data(test_solution_data)
    
    X_test = test_data['text']
    y_true = test_solution_data['GENRE']
    
    # Vectorize test data
    X_test_vectorized = vectorizer.transform(X_test)
    
    # Predict genres for the test data
    y_pred = model.predict(X_test_vectorized)
    
    # Encode the predicted labels
    y_pred_encoded = label_encoder.inverse_transform(y_pred)
    
    # Create a DataFrame for predictions
    test_solution_data['Predicted_GENRE'] = y_pred_encoded
    
    # Merge predictions with test_solution_data to check accuracy
    merged_df = test_solution_data[['ID', 'GENRE', 'Predicted_GENRE']]
    merged_df = merged_df.rename(columns={'GENRE': 'True_GENRE', 'Predicted_GENRE': 'Predicted_GENRE'})
    
    # Print predictions
    print("ID\tTrue Genre\tPredicted Genre")
    print("-" * 40)
    for _, row in merged_df.iterrows():
        print(f"{row['ID']}\t{row['True_GENRE']}\t{row['Predicted_GENRE']}")
    
    # Calculate number of matches
    num_matches = (merged_df['True_GENRE'] == merged_df['Predicted_GENRE']).sum()
    total_predictions = len(merged_df)
    accuracy = num_matches / total_predictions if total_predictions > 0 else "N.A"

    if accuracy != "N.A":
        print(f"\nModel Accuracy: {accuracy:.2f}")
        print(f"Number of Correct Predictions: {num_matches}")
    else:
        print("\nSome predictions are not available. Accuracy: N.A")

if __name__ == "__main__":
    main()
