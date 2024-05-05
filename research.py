import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load data from CSV file
df = pd.read_csv('anemia.csv')

# Select columns to use as features
feature_columns = ['Glucose', 'BloodPressure', 'Insulin', 'BMI', 'AnemiaPedigreeFunction', 'Age']

# Select the target column
target_column = 'Outcome'

# Split the dataset into features and target variable
X = df[feature_columns]
y = df[target_column]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train Gaussian NB
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)
nb_y_pred = nb_model.predict(X_test_scaled)
nb_accuracy_new = accuracy_score(y_test, nb_y_pred)
nb_accuracy_original = 0.84  # Original accuracy for NB

# initialize and train SVM
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_scaled, y_train)
svm_y_pred = svm_model.predict(X_test_scaled)
svm_accuracy_new = accuracy_score(y_test, svm_y_pred)
svm_accuracy_original = 0.89  # Original accuracy for SVM

# intialize and train knn model
knn_model = KNeighborsClassifier()
knn_model.fit(X_train_scaled, y_train)
knn_y_pred = knn_model.predict(X_test_scaled)
knn_accuracy_new = accuracy_score(y_test, knn_y_pred)
knn_accuracy_original = 0.12  # Original accuracy for KNN

# Storing accuracies in dictionaries
new_scores = {
    "Gaussian Naive Bayes": nb_accuracy_new,
    "SVM": svm_accuracy_new,
    "KNN": knn_accuracy_new
}

original_scores = {
    "Gaussian Naive Bayes": nb_accuracy_original,
    "SVM": svm_accuracy_original,
    "KNN": knn_accuracy_original
}

# Plotting the bar graph
labels = new_scores.keys()
x = range(len(labels))

plt.figure(figsize=(12, 6))

plt.bar(x, new_scores.values(), width=0.4, align='center', label='New Scores', color='blue')
plt.bar([i + 0.4 for i in x], original_scores.values(), width=0.4, align='center', label='Original Scores', color='red')

plt.xlabel('Classifiers')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison')
plt.xticks([i + 0.2 for i in x], labels)
plt.ylim(0, 1)  # Set y-axis limit from 0 to 1
plt.legend()
plt.show()
