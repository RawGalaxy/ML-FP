import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA

# Load the dataset
data = pd.read_csv('C:/Users/ariqj/OneDrive/Documents/University/Machine Learning/SVM/winequality-red.csv')

# Use more features based on correlation and feature importance
features = ['sulphates', 'alcohol', 'citric acid', 'volatile acidity']
X = data[features]
y = (data['quality'] > 5).astype(int)  # Using integer labels

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle imbalanced dataset
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Display the dataset after SMOTE
print("Features after SMOTE resampling:")
print(X_train_smote.head())  # Show the first few rows of the resampled features
print("\nLabels distribution after SMOTE resampling:")
print(y_train_smote.value_counts())  # Show the distribution of the resampled labels

# Preprocessing pipeline with PCA
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=2))  # Reduce to 2 components for visualization
        ]), features)
    ])

# Create a pipeline with the classifier
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', SVC())
])

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__gamma': [1, 0.1, 0.01, 0.001],
    'classifier__kernel': ['rbf', 'poly', 'linear']
}

grid = GridSearchCV(pipeline, param_grid, refit=True, verbose=2, cv=5)
grid.fit(X_train_smote, y_train_smote)

# Using the best estimator
best_model = grid.best_estimator_

# Evaluate the model
accuracy = best_model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

# Get classification report
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))

# Plotting the decision boundary for the selected features after PCA
X_train_transformed = best_model.named_steps['preprocessor'].transform(X_train_smote)

plt.figure(figsize=(10, 6))
DecisionBoundaryDisplay.from_estimator(
    best_model.named_steps['classifier'],
    X_train_transformed,
    response_method="predict",
    cmap=plt.cm.Spectral,
    alpha=0.8,
    xlabel='Principal Component 1',
    ylabel='Principal Component 2'
)
plt.scatter(X_train_transformed[:, 0], 
            X_train_transformed[:, 1], 
            c=y_train_smote, s=20, edgecolor="k")
plt.title('SVM Decision Boundary with PCA')
plt.show()
