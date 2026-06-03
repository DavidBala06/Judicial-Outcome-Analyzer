from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np 
import data_preparation as dp
import pandas as pd 

rf_model = RandomForestClassifier(n_estimators = 100, random_state = 42, class_weight="balanced")
rf_model.fit(dp.X_train, dp.y_train)

y_pred = rf_model.predict(dp.X_test)

accuracy = accuracy_score(dp.y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(dp.y_test, y_pred, target_names = dp.target_classes))

# Confusion Matrix
plt.figure(figsize = (8,6))
cm = confusion_matrix(dp.y_test, y_pred)    
sns.heatmap(cm, annot = True, fmt = "d", cmap = "Blues", cbar = False)
plt.show()

# Feature Importance
feature_importances = rf_model.feature_importances_
feature_names = dp.X_encoded.columns

feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
feature_df = feature_df.sort_values(by='Importance', ascending=False).head(10) # Top 10

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_df, x='Importance', y='Feature', palette='viridis')
plt.title('Top 10 Most Important Factors Influencing the Verdict', fontsize=14, pad=15)
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.show()
