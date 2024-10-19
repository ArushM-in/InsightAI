import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample




data_path = 'b_depressed.csv'
data = pd.read_csv(data_path)

# Fill missing values
data = data.fillna(data.mean())

# Define features and label
X = data.drop(columns=['depressed'])  # Features
y = data['depressed']  # Target variable

# Balance the dataset
# Majority and minority classes
df_majority = data[data['depressed'] == 0]
df_minority = data[data['depressed'] == 1]

# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                  replace=True,     # Sample with replacement
                                  n_samples=len(df_majority),    # Match majority class
                                  random_state=42) # Reproducible results


data_balanced = pd.concat([df_majority, df_minority_upsampled])

# Define features and label again
X_balanced = data_balanced.drop(columns=['depressed'])
y_balanced = data_balanced['depressed']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Create a Random Forest model
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Get predicted probabilities
y_proba = model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class


threshold = 0.6  # ADJUSTABLE
y_pred = (y_proba >= threshold).astype(int)  

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Print classification report
print(classification_report(y_test, y_pred))

# Print confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)


print(y.value_counts())