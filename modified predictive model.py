import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample
import pandas as pd
import numpy as np


# Load your data
data_path = 'modified_data.csv'
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

# Combine majority class with upsampled minority class
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

# Set a custom threshold
threshold = 0.6  # You can adjust this value
y_pred = (y_proba >= threshold).astype(int)  # Predict based on the custom threshold

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

# def predict_depression(user_input):
#     # Convert user input to DataFrame
#     input_data = pd.DataFrame([user_input])

#     # Fill missing values (just like during training)
#     input_data = input_data.fillna(X.mean())  # Fill missing values with the mean

#     # Make sure the input data has the same columns as X
#     input_data = input_data.reindex(columns=X.columns, fill_value=0)

#     # Predict probabilities
#     probabilities = model.predict_proba(input_data)[:, 1]  # Get probability of class 1 (depressed)
    
#     return probabilities[0]

# # Now, you can ask for user input
# while True:
#     # Gather input from the user
#     user_input = {}
#     user_input['sex'] = int(input("Enter sex (0 or 1): "))
#     user_input['Age'] = float(input("Enter Age: "))
#     user_input['Married'] = int(input("Enter Married status (0 or 1): "))
#     user_input['Number_children'] = int(input("Enter Number of children: "))
#     user_input['education_level'] = int(input("Enter education level (numeric): "))
#     user_input['total_members'] = int(input("Enter total members in household: "))
#     user_input['gained_asset'] = float(input("Enter gained asset: "))
#     user_input['durable_asset'] = float(input("Enter durable asset: "))
#     user_input['save_asset'] = float(input("Enter save asset: "))
#     user_input['living_expenses'] = float(input("Enter living expenses: "))
#     user_input['other_expenses'] = float(input("Enter other expenses: "))
#     user_input['incoming_salary'] = float(input("Enter incoming salary: "))
#     user_input['incoming_own_farm'] = float(input("Enter income from own farm: "))
#     user_input['incoming_business'] = float(input("Enter income from business: "))
#     user_input['incoming_no_business'] = float(input("Enter income from no business: "))
#     user_input['incoming_agricultural'] = float(input("Enter income from agricultural: "))
#     user_input['farm_expenses'] = float(input("Enter farm expenses: "))
#     user_input['labor_primary'] = int(input("Enter labor primary (0 or 1): "))
#     user_input['lasting_investment'] = float(input("Enter lasting investment: "))
#     user_input['no_lasting_investmen'] = float(input("Enter non-lasting investment: "))

#     # Get prediction
#     likelihood = predict_depression(user_input)
#     print(f"The likelihood of being depressed is: {likelihood:.2f}")

#     # Ask if the user wants to continue
#     cont = input("Do you want to input another data? (yes/no): ")
#     if cont.lower() != 'yes':
#         break