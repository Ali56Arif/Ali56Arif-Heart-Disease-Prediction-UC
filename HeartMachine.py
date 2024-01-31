Loading the dataset
file_path = '/kaggle/input/heart-disease-prediction-uci/heart.csv' heart_data = pd.read_csv(file_path)

Kala saarida sifooyinka iyo bartilmaameedka
X = heart_data.drop('target', axis=1) y = heart_data['target']

Qaybinta xogta tababarka iyo tijaabada
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

Abuurista iyo tababarka modelka 'Random Forest Classifier'
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42) rf_classifier.fit(X_train, y_train)

Saadaalinta xogta tijaabada
y_pred = rf_classifier.predict(X_test)

Qiimeynta modelka
accuracy = accuracy_score(y_test, y_pred) print(f"Saxnaanta Modelka: {accuracy}") print(classification_report(y_test, y_pred))

