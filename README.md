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


Marka aad heshay natiijooyinka tababarka iyo qiimeynta modelkaaga 'Random Forest Classifier', waxaa jira dhowr talaabo oo aad qaadi karto si aad u sii horumariso ama u dabaqdo modelkaaga:

1. **Hagaajinta Modelka**: Haddii aad rabto inaad sii hagaajiso waxqabadka modelka, waxaad isku dayi kartaa:
   - Isbeddelka xuduudaha modelka, sida `n_estimators`, `max_depth`, `min_samples_split`, iwm.
   - Isticmaalka feature engineering si aad u abuurto ama u doorato sifooyinka saameynta ugu badan ku leh saadaasha.
   - Tijaabinta hababka kala duwan ee xogta loo nadiifiyo ama loo beddelo.

2. **Cross-Validation**: Isticmaal cross-validation si aad u hubiso in modelkaagu si joogto ah u shaqeynayo oo uusan ku dhicin xad-dhaaf ama xad-gudub hoose.

3. **Tijaabinta Modello Kale**: Haddii aad rabto inaad aragto sida modello kale u shaqeeyaan xogtaada, waxaad tijaabin kartaa modello kala duwan sida 'Logistic Regression', 'Support Vector Machines', ama 'Gradient Boosting Classifiers'.

4. **Fahamka Waxqabadka Modelka**: Falanqee natiijooyinka si aad u fahamto meelaha uu modelkaagu ku xooggan yahay iyo meelaha uu ka liito. Tani waxaa ka mid noqon kara fahamka muhiimadda sifooyinka iyo sidoo kale falanqeynta qaladaadka noocyada kala duwan (sida qaladaadka nooca koowaad iyo labaad).

5. **Dabaqidda Modelka Xaaladaha Dhabta ah**: Haddii aad ku qanacsan tahay waxqabadka modelka, waxaad tijaabin kartaa inaad ku dabaqdo xaalado dhab ah ama xog cusub oo aan horay loo arag.

6. **Kaydinta iyo Dib-u-isticmaalka Modelka**: Kaydi modelka si aad ugu isticmaasho mustaqbalka iyadoo aan loo baahnayn in mar kale la tababaro. Tani waxay faa'iido u leedahay haddii aad rabto inaad ku dabaqdo modelka xog cusub.

7. **Warbixinta iyo Bandhigidda Natiijooyinka**: Haddii loo baahdo, diyaari warbixinno ama bandhigyo ku saabsan waxqabadka modelka iyo natiijooyinka aad ka heshay, gaar ahaan haddii mashruucani yahay qayb ka mid ah cilmi-baaris ama hawlo ganacsi.

Xusuusnow, mar walba waa muhiim in la tixgeliyo arrimaha sida saxda ah ee modelka loo dabaqayo iyo in la hubiyo inuu si cadaalad ah u shaqeynayo, gaar ahaan marka la isticmaalayo xog xasaasi ah sida xogta caafimaadka.
