# # import pandas as pd
# # from transformers import GPT2LMHeadModel, GPT2Tokenizer


# # url = "https://www.kaggle.com/datasets/diegobabativa/depression/download"
# # # Change the path according to where you downloaded the dataset
# # data_path = 'b_depressed.csv'  # Adjust this to your actual file path

# # # Read the dataset
# # data = pd.read_csv(data_path)

# # # Fill missing values with the mean of each column
# # data = data.fillna(data.mean())

# # # Display the first few rows of the dataset
# # #print(data.head())

# # # Define features and label
# # X = data.drop(columns=['depressed'])  # Features (all columns except 'depressed')
# # y = data['depressed']  # Target variable


# # from sklearn.model_selection import train_test_split
# # from sklearn.linear_model import LogisticRegression
# # from sklearn.metrics import accuracy_score

# # # Split the data into training and testing sets
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # # Create a Logistic Regression model
# # model = LogisticRegression(max_iter=1000)  # Increase max_iter if you encounter convergence issues

# # # Train the model
# # model.fit(X_train, y_train)

# # # Make predictions
# # predictions = model.predict(X_test)

# # # Calculate accuracy
# # accuracy = accuracy_score(y_test, predictions)
# # print(f'Accuracy: {accuracy:.2f}')

# # from sklearn.metrics import classification_report, confusion_matrix

# # # Print classification report
# # print(classification_report(y_test, predictions))

# # # Print confusion matrix
# # cm = confusion_matrix(y_test, predictions)
# # print("Confusion Matrix:")
# # print(cm)
# # print(y.value_counts())

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from imblearn.over_sampling import SMOTE

# # Load data and fill missing values as before
# data = pd.read_csv('b_depressed.csv')
# data = data.fillna(data.mean())

# X = data.drop(columns=['depressed'])
# y = data['depressed']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize SMOTE
# smote = SMOTE(random_state=42)

# # Apply SMOTE to the training data
# X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# # Create a Logistic Regression model
# model = LogisticRegression(max_iter=1000)

# # Train the model on the resampled data
# model.fit(X_resampled, y_resampled)

# # Make predictions on the test set
# predictions = model.predict(X_test)

# # Calculate accuracy
# accuracy = accuracy_score(y_test, predictions)
# print(f'Accuracy: {accuracy:.2f}')

# # Print classification report and confusion matrix
# print(classification_report(y_test, predictions))
# cm = confusion_matrix(y_test, predictions)
# print("Confusion Matrix:")
# print(cm)

import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from imblearn.over_sampling import SMOTE  # Import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

url = "https://www.kaggle.com/datasets/diegobabativa/depression/download"

data_path = 'b_depressed.csv'  


data = pd.read_csv(data_path)

data = data.fillna(data.mean())


X = data.drop(columns=['depressed'])  # Features (all columns except 'depressed')
y = data['depressed']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to the training dataF
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Create a Logistic Regression model
model = LogisticRegression(max_iter=1000)  # Increase max_iter if you encounter convergence issues

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

# Print value counts for depressed class
print(y.value_counts())