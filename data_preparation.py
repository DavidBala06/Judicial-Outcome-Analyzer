import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('predictive_criminal_justice_dataset.csv')
df_clean = df.drop("Case_ID", axis=1) #Case Id is not useful for prediction, so we drop it.

X = df_clean.drop("Case_Outcome", axis = 1)
y = df_clean["Case_Outcome"]

# Encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

target_classes = label_encoder.classes_
print(f"\n Target Classes: {target_classes}")
categorical_columns = X.select_dtypes(include = ["object"]).columns
X_encoded = pd.get_dummies(X, columns = categorical_columns, drop_first = True)

print(f"Dimensions of the feature set: {X_encoded.shape}")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y_encoded,
    test_size = 0.2,
    random_state = 42,
    stratify = y_encoded
)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")
