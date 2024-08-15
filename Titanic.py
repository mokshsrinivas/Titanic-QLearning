import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the training and test datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Preprocess the data
def preprocess(data, is_train=True):
    # Fill missing values with the mean for numeric columns
    data['Age'].fillna(data['Age'].mean(), inplace=True)
    data['Fare'].fillna(data['Fare'].mean(), inplace=True)
    
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
    
    # Encode categorical features
    label_encoders = {}
    categorical_features = ['Sex', 'Embarked']
    for feature in categorical_features:
        le = LabelEncoder()
        data[feature] = le.fit_transform(data[feature])
        label_encoders[feature] = le

    # Select features and target
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    if is_train:
        return data[features].values, data['Survived'].values, label_encoders
    else:
        return data[features].values, data['PassengerId'].values, label_encoders

# Load the training and test datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Preprocess the training and test data
X_train, y_train, label_encoders = preprocess(train_data)
X_test, passenger_ids, _ = preprocess(test_data, is_train=False)

# Initialize Q-table
state_space_size = len(X_train)  # Number of unique states in training data
action_space = [0, 1]  # 0: Did not survive, 1: Survived
q_table = np.zeros((state_space_size, len(action_space)))

# Map each state (row of X_train) to a unique index
state_to_index = {tuple(state): idx for idx, state in enumerate(X_train)}

# Hyperparameters
alpha = 0.05  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.99  # Decay rate for exploration
num_epochs = 10000  # Number of training epochs

# Training loop
for epoch in range(num_epochs):
    for i, state in enumerate(X_train):
        state_index = state_to_index[tuple(state)]
        if np.random.rand() < epsilon:
            action = np.random.choice(action_space)  # Explore
        else:
            action = np.argmax(q_table[state_index])  # Exploit
        
        # Assume we get a reward based on the correctness of the action
        actual_survival = y_train[i]
        reward = 1 if action == actual_survival else -1
        
        # Update Q-value
        q_table[state_index, action] = q_table[state_index, action] + alpha * (
            reward + gamma * np.max(q_table[state_index]) - q_table[state_index, action]
        )
    
    # Decay epsilon
    epsilon *= epsilon_decay

# Function to predict using the Q-table
def predict(state):
    state_tuple = tuple(state)
    if state_tuple in state_to_index:
        state_index = state_to_index[state_tuple]
        return np.argmax(q_table[state_index])
    else:
        # If the state is unseen, assign a default Q-value (mean of Q-values)
        return np.argmax(np.mean(q_table, axis=0))

# Make predictions on the test data
predictions = [predict(state) for state in X_test]

# Create a DataFrame for the output
output = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': predictions})

# Save the predictions to a CSV file
output.to_csv('predictions.csv', index=False)