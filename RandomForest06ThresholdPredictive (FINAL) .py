import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample
import pandas as pd
import numpy as np



data_path = 'b_depressed.csv'
data = pd.read_csv(data_path)

# Fill missing values
data = data.fillna(data.mean())


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

# ADJUSTABLE
threshold = 0.6  
y_pred = (y_proba >= threshold).astype(int)  

#  accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Print classification report
print(classification_report(y_test, y_pred))

# Print confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)


print(y.value_counts())

def predict_depression(user_input):
    # dataframe conversion
    input_data = pd.DataFrame([user_input])

    #
    input_data = input_data.fillna(X.mean())  # Fill missing values with the mean

    
    input_data = input_data.reindex(columns=X.columns, fill_value=0)

    
    probabilities = model.predict_proba(input_data)[:, 1]  # Get probability of class 1 (depressed)
    
    return probabilities[0]

#input
while True:
    #INPUT
    user_input = {}
    user_input['sex'] = int(input("Enter sex (0: Female or 1: Male): "))
    user_input['Age'] = float(input("Enter Age: "))
    user_input['Married'] = int(input("Enter Married status (0: Unmarried or 1:Married): "))
    user_input['Number_children'] = int(input("Enter Number of children: "))
    user_input['education_level'] = int(input("Enter education level (1- 15): "))
    user_input['total_members'] = int(input("Enter total members in household: "))
    user_input['gained_asset'] = float(input("Enter gained asset in cents: "))
    user_input['durable_asset'] = float(input("Enter durable asset in cents: "))
    user_input['save_asset'] = float(input("Enter saved asset in cents: "))
    user_input['living_expenses'] = float(input("Enter annual living expenses in cents: "))
    user_input['other_expenses'] = float(input("Enter other annual expenses in cents : "))
    user_input['incoming_salary'] = float(input("Enter incoming salary (Binary): "))
    user_input['incoming_own_farm'] = float(input("Enter income from own farm (Binary): "))
    user_input['incoming_business'] = float(input("Enter income from business (Binary): "))
    user_input['incoming_no_business'] = float(input("Enter income from no business (Binary): "))
    user_input['incoming_agricultural'] = float(input("Enter annual income from agricultural in cents: "))
    user_input['farm_expenses'] = float(input("Enter annual farm expenses: "))
    user_input['labor_primary'] = int(input("Enter labor primary (0 or 1): "))
    user_input['lasting_investment'] = float(input("Enter lasting investment in cents: "))
    user_input['no_lasting_investmen'] = float(input("Enter non-lasting investment in cents: "))

    # Get prediction
    likelihood = predict_depression(user_input)
    print(f"The likelihood of being depressed is: {likelihood:.2f}")
    print(f"A value above 0.6 indicates a recommendation of talking to a proffessional. Note these values may be innacurate.")

   
    cont = input("Do you want to input another data? (yes/no): ")
    if cont.lower() != 'yes':
        break
