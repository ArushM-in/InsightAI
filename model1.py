import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer


url = "https://www.kaggle.com/datasets/diegobabativa/depression/download"

data_path = 'b_depressed.csv'  


data = pd.read_csv(data_path)


data = data.fillna(data.mean())


#print(data.head())


X = data.drop(columns=['depressed'])  # Features (all columns except 'depressed')
y = data['depressed']  # Target variable


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Logistic Regression model
model = LogisticRegression(max_iter=1000)  

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy:.2f}')

from sklearn.metrics import classification_report, confusion_matrix

# Print classification report
print(classification_report(y_test, predictions))

# Print confusion matrix
cm = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(cm)

print(y.value_counts())
