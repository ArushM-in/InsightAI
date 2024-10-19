import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE


data_path = 'b_depressed.csv'  
data = pd.read_csv(data_path)

# Fill missing values with the mean of each column
data = data.fillna(data.mean())


X = data.drop(columns=['depressed'])  # Features (all columns except 'depressed')
y = data['depressed']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Check class distribution after resampling
print(pd.Series(y_train_resampled).value_counts())


model = RandomForestClassifier(random_state=42)

# Train the model on the resampled data
model.fit(X_train_resampled, y_train_resampled)

# Make predictions
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy:.2f}')

# Print classification report
print(classification_report(y_test, predictions))

# Print confusion matrix
cm = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(cm)


print("Original class distribution:")
print(y.value_counts())
