Modelka 'Random Forest Classifier' ee aan tababarnay waxa uu leeyahay saxnaan ah 82.42%. Warbixinta classification-ka waxa ay bixisaa faahfaahin dheeraad ah oo ku saabsan waxqabadka modelka:

- **Precision**: Qiyaasta saxda ah ee saadaalinta togan. Tusaale ahaan, precision-ka saadaalinta xaaladda wadnaha ('1') waa 84%, taasoo macnaheedu yahay in 84% saadaalinta wadnaha ee modelku uu sameeyay ay ahaayeen kuwo sax ah.
- **Recall**: Waa saamiga saadaalinta togan ee dhab ahaantii sax ah. Tusaale ahaan, recall-ka saadaalinta wadnaha ('1') waa 84%, taasoo tilmaamaysa in modelku uu si sax ah u qabtay 84% kiisaska dhabta ah ee wadnaha.
- **F1-Score**: Waa celceliska harmonic ee precision iyo recall. Waa qiyaas wanaagsan oo isu dheellitirnaanta iyo waxtarka guud ee modelka.
- **Accuracy**: Waa saamiga saadaalinta saxda ah ee dhammaan saadaalinta. Halkan, waxay tahay 82.42%.

Guud ahaan, modelku waxa uu muujiyey waxqabad wanaagsan oo isu dheelitiran. Haddii loo baahdo,
waxaa suurtagal ah in la sameeyo hagaajinno dheeraad ah, sida hagaajinta xuduudaha modelka ama isku dayga hababka kala duwan ee feature engineering, 
si loo hagaajiyo waxqabadka modelka. Sidoo kale, waxaa muhiim ah in la tixgeliyo in modelka la tijaabiyo iyadoo la adeegsanayo xog ka duwan si loo hubiyo inuu si wanaagsan u guud ahaan yahay.

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Loading the dataset
file_path = '/kaggle/input/heart-disease-prediction-uci/heart.csv'
heart_data = pd.read_csv(file_path)

# Kala saarida sifooyinka iyo bartilmaameedka
X = heart_data.drop('target', axis=1)
y = heart_data['target']

# Qaybinta xogta tababarka iyo tijaabada
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Abuurista iyo tababarka modelka 'Random Forest Classifier'
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Saadaalinta xogta tijaabada
y_pred = rf_classifier.predict(X_test)

# Qiimeynta modelka
accuracy = accuracy_score(y_test, y_pred)
print(f"Saxnaanta Modelka: {accuracy}")
print(classification_report(y_test, y_pred))
![image](https://github.com/Ali56Arif/Ali56Arif-Heart-Disease-Prediction-UC/assets/79138028/c5ac9df3-c0ae-4be8-9a95-a06a15141cf1)

Halkan Ka eeg
https://www.kaggle.com/code/ali56arif/heart-d
