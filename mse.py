import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load data
file_path = 'credit_data.csv'
data = pd.read_csv(file_path)

# Features and target variable
X = data[['Age', 'Income', 'LoanAmount']]
y = data['CreditScore']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display results
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Predict function
def predict_credit_score(age, income, loan_amount):
    prediction = model.predict([[age, income, loan_amount]])
    return prediction[0]

# Example prediction



#Taking user input for prediction
ag = int(input("Enter Your Age"))
inc = int(input("Enter Your Income"))
lamt = int(input("Enter Your Loan Amount"))
credit_prediction = predict_credit_score(ag, inc, lamt)
print("Predicted Credit Score:", credit_prediction)
