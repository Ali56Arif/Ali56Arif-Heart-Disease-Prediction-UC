Waxaan isticmaali karnaa dataset-ka cusub ee 'heart.csv' si aan u tababarno model. Aan marka hore eegno waxa ku jira dataset-kan cusub ka dibna aan sameyno falanqeyn iyo tababar model.

Waxaan billaabayaa adigoo faylka xogta akhrinaya si aan u fahmo qaab dhismeedkiisa iyo nooca xogta uu ka kooban yahay. Kadibna, waxaan bixin doonaa talooyin ku saabsan sida loo isticmaalo xogtan si loo tababaro model Machine Learning ah. Aan ku bilaabo koodhkan:

Dataset-ka 'heart.csv' waxa uu ka kooban yahay xog ku saabsan bukaanada iyo inay qabaan xaalad wadnaha ah (lagu muujiyey tiirka 'target'). Sifooyinka (features) waxaa ka mid ah:

age: Da'da qofka.
sex: Jinsiga qofka (1 = Rag, 0 = Dumar).
cp: Nooca xanuunka laabta (chest pain type).
trestbps: Cadaadiska dhiigga ee nasashada (resting blood pressure).
chol: Heerka kolestaroolka dhiigga.
fbs: Heerka sonkorta dhiigga marka la soomo (fasting blood sugar > 120 mg/dl).
restecg: Natiijooyinka electrocardiographic ee nasashada.
thalach: Garaaca wadnaha ugu sarreeya.
exang: Xanuunka laabta ee jimicsiga (exercise induced angina).
oldpeak: ST depression ee ka dhalatay jimicsiga marka loo eego nasashada.
slope: Slope-ka ST segment exercise peak.
ca: Tirada weyn ee xididdada wadnaha ee la arki karo.
thal: Cilad (3 = normal; 6 = fixed defect; 7 = reversible defect).
Tiirka target waa bartilmaameedka (target variable), kaasoo muujinaya haddii qofku qabo xaalad wadnaha ah (1 = haa, 0 = maya).

Haa, waxaan isticmaali karnaa dataset-ka cusub ee 'heart.csv' si aan u tababarno model. Dataset-kan waa mid ka yar oo laga yaabo inuu ka fudud yahay in la maareeyo marka loo eego dataset-ka duumada ee weynaa. Aan marka hore eegno waxa ku jira dataset-kan cusub ka dibna aan sameyno falanqeyn iyo tababar model. 

Waxaan billaabayaa adigoo faylka xogta akhrinaya si aan u fahmo qaab dhismeedkiisa iyo nooca xogta uu ka kooban yahay. Kadibna, waxaan bixin doonaa talooyin ku saabsan sida loo isticmaalo xogtan si loo tababaro model Machine Learning ah. Aan ku bilaabo koodhkan:

Dataset-ka 'heart.csv' waxa uu ka kooban yahay xog ku saabsan bukaanada iyo inay qabaan xaalad wadnaha ah (lagu muujiyey tiirka 'target'). Sifooyinka (features) waxaa ka mid ah:

- `age`: Da'da qofka.
- `sex`: Jinsiga qofka (1 = Rag, 0 = Dumar).
- `cp`: Nooca xanuunka laabta (chest pain type).
- `trestbps`: Cadaadiska dhiigga ee nasashada (resting blood pressure).
- `chol`: Heerka kolestaroolka dhiigga.
- `fbs`: Heerka sonkorta dhiigga marka la soomo (fasting blood sugar > 120 mg/dl).
- `restecg`: Natiijooyinka electrocardiographic ee nasashada.
- `thalach`: Garaaca wadnaha ugu sarreeya.
- `exang`: Xanuunka laabta ee jimicsiga (exercise induced angina).
- `oldpeak`: ST depression ee ka dhalatay jimicsiga marka loo eego nasashada.
- `slope`: Slope-ka ST segment exercise peak.
- `ca`: Tirada weyn ee xididdada wadnaha ee la arki karo.
- `thal`: Cilad (3 = normal; 6 = fixed defect; 7 = reversible defect).

Tiirka `target` waa bartilmaameedka (target variable), kaasoo muujinaya haddii qofku qabo xaalad wadnaha ah (1 = haa, 0 = maya).
import pandas as pd

# Loading the heart dataset
file_path = '/mnt/data/heart.csv'
heart_data = pd.read_csv(file_path)

# Displaying the first few rows of the dataset to understand its structure and content
heart_data.head()

### Tallaabooyinka Tababarka Modelka:

1. **Falanqeynta iyo Nadiifinta Xogta**: Hubi in xogtu nadiif tahay oo aanay lahayn wax qiimayaal maqan ah ama qaladaad.

2. **Kala Qaybinta Xogta**: Qaybi xogta tababarka iyo tijaabada.

3. **Dooro Modelka**: Maadaama aan hayno xog classification ah, waxaan dooran karnaa model sida 'Random Forest Classifier', 'Logistic Regression', ama mid kale oo ku habboon classification.

4. **Tababarka iyo Qiimeynta Modelka**: Tababar modelka oo qiimee waxqabadkiisa adigoo isticmaalaya metrikada sida saxnaanta, F1 score, iwm.

Aan ku billaabo hirgelinta tallaabooyinkan adigoo isticmaalaya mid ka mid ah modelada.

