#iris
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# # Ensemble Libraries
# from sklearn.ensemble import RandomForestClassifier, StackingClassifier
# from xgboost import XGBClassifier

# # Stacking Model Libraries
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier

# # 1. Load and split the dataset
# iris = load_iris()
# X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# # --- Method 1: Bagging (Random Forest) ---
# print("Running Bagging (Random Forest)...")
# bagging_model = RandomForestClassifier(n_estimators=100, random_state=42)
# bagging_model.fit(X_train, y_train)
# y_pred_bagging = bagging_model.predict(X_test)
# print(f"Bagging Accuracy: {accuracy_score(y_test, y_pred_bagging)}\n")


# # --- Method 2: Boosting (XGBoost) ---
# print("Running Boosting (XGBoost)...")
# boosting_model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
# boosting_model.fit(X_train, y_train)
# y_pred_boosting = boosting_model.predict(X_test)
# print(f"Boosting Accuracy: {accuracy_score(y_test, y_pred_boosting)}\n")


# # --- Method 3: Stacking ---
# print("Running Stacking Classifier...")
# # Define the base models
# base_models = [
#     ('knn', KNeighborsClassifier(n_neighbors=3)),
#     ('dt', DecisionTreeClassifier(max_depth=3))
# ]
# # Define the meta-model
# meta_model = SVC(kernel='linear', probability=True)

# # Train the Stacking Classifier
# stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model)
# stacking_model.fit(X_train, y_train)
# y_pred_stacking = stacking_model.predict(X_test)
# print(f"Stacking Accuracy: {accuracy_score(y_test, y_pred_stacking)}\n")

#---------------------------------------------------------------------------------------------

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.metrics import accuracy_score
# from sklearn.ensemble import RandomForestClassifier, StackingClassifier
# from xgboost import XGBClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier

# # --- 1. Load dataset ---
# file_path = "your_dataset.csv"  # Replace with your CSV file
# df = pd.read_csv(file_path)

# # --- 2. Preliminary cleaning ---
# # Drop rows with all NaNs
# df.dropna(how='all', inplace=True)

# # Fill remaining NaNs with column mean (numeric) or mode (categorical)
# for col in df.columns:
#     if df[col].dtype == object:
#         df[col].fillna(df[col].mode()[0], inplace=True)
#     else:
#         df[col].fillna(df[col].mean(), inplace=True)

# # Encode categorical columns
# label_encoders = {}
# for col in df.columns:
#     if df[col].dtype == object:
#         le = LabelEncoder()
#         df[col] = le.fit_transform(df[col])
#         label_encoders[col] = le

# # --- 3. Split features and target ---
# # Assuming last column is target; modify if different
# X = df.iloc[:, :-1]
# y = df.iloc[:, -1]

# # Optional: scale features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# # --- 4. Bagging (Random Forest) ---
# bagging_model = RandomForestClassifier(n_estimators=100, random_state=42)
# bagging_model.fit(X_train, y_train)
# y_pred_bagging = bagging_model.predict(X_test)
# print(f"Bagging Accuracy: {accuracy_score(y_test, y_pred_bagging):.4f}")

# # --- 5. Boosting (XGBoost) ---
# boosting_model = XGBClassifier(n_estimators=100, learning_rate=0.1, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
# boosting_model.fit(X_train, y_train)
# y_pred_boosting = boosting_model.predict(X_test)
# print(f"Boosting Accuracy: {accuracy_score(y_test, y_pred_boosting):.4f}")

# # --- 6. Stacking ---
# base_models = [
#     ('knn', KNeighborsClassifier(n_neighbors=3)),
#     ('dt', DecisionTreeClassifier(max_depth=3))
# ]
# meta_model = LogisticRegression(max_iter=1000)
# stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model)
# stacking_model.fit(X_train, y_train)
# y_pred_stacking = stacking_model.predict(X_test)
# print(f"Stacking Accuracy: {accuracy_score(y_test, y_pred_stacking):.4f}")
