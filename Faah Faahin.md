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
file_path = 'https://github.com/Ali56Arif/Ali56Arif-Heart-Disease-Prediction-UC/blob/main/heart.csv'
heart_data = pd.read_csv(file_path)

# Displaying the first few rows of the dataset to understand its structure and content
heart_data.head()

### Tallaabooyinka Tababarka Modelka:

1. **Falanqeynta iyo Nadiifinta Xogta**: Hubi in xogtu nadiif tahay oo aanay lahayn wax qiimayaal maqan ah ama qaladaad.

2. **Kala Qaybinta Xogta**: Qaybi xogta tababarka iyo tijaabada.

3. **Dooro Modelka**: Maadaama aan hayno xog classification ah, waxaan dooran karnaa model sida 'Random Forest Classifier', 'Logistic Regression', ama mid kale oo ku habboon classification.

4. **Tababarka iyo Qiimeynta Modelka**: Tababar modelka oo qiimee waxqabadkiisa adigoo isticmaalaya metrikada sida saxnaanta, F1 score, iwm.

Aan ku billaabo hirgelinta tallaabooyinkan adigoo isticmaalaya mid ka mid ah modelada.

Aan isticmaalno 'Random Forest Classifier' si aan u tababarno oo aan u qiimeyno model ku salaysan dataset-ka wadnaha. Waxaan raaci doonaa tallaabooyinka soo socda:

Diyaarinta Xogta: Nadiifinta xogta haddii loo baahdo.

Kala Qaybinta Xogta: Qaybinta xogta u dhexeysa xogta tababarka iyo xogta tijaabada.

Tababarka Modelka 'Random Forest Classifier': Isticmaalka xogta tababarka si loo tababaro modelka.

Qiimeynta Modelka: Isticmaalka xogta tijaabada si loo qiimeeyo waxqabadka modelka.

Si aad ugu dabaqdo modelkaaga 'Random Forest Classifier' cusbitaalka, tusaale Cusbitaalka magaalada Burco, waxaa muhiim ah inaad raacdo dhowr talaabo oo lagama maarmaan ah si loo hubiyo in modelkaagu si wax ku ool ah u shaqeynayo isla markaana u adeegsan karo baahida caafimaad ee deegaankaas:

1. **Fahamka Xogta Cusub**: Hubi in xogta laga heli karo cusbitaalka magaalada Burco ay la jaan qaadi karto qaabka iyo nooca xogta aad u adeegsatay tababarka modelkaaga. Tani waxaa ka mid noqon kara in la heli karo isla sifooyinka caafimaad ee aad isticmaashay.

2. **Tijaabinta Modelka Xogta Dhabta ah**: Kahor inta aanad si buuxda u dabaqin modelka, waa muhiim inaad ku tijaabiso xog yar oo ka timid cusbitaalka si aad u aragto sida uu uga jawaabayo xaalado dhab ah. Tani waxay kaa caawin doontaa inaad ogaato haddii ay jiraan wax hagaajin ama wax ka beddel ah oo loo baahan yahay.

3. **Tababarka Shaqaalaha Cusbitaalka**: Shaqaalaha caafimaadka waa inay fahmaan sida loo isticmaalo modelka. Waxaa muhiim ah in la siiyo tababar iyo hagid ku saabsan sida loo geliyo xogta, sida loo fasirto natiijooyinka, iyo waxa la sameeyo haddii ay jiraan su'aalo ama arrimo.

4. **La Socodka iyo Qiimeynta Joogtada ah**: Marka modelka la hirgeliyo, waa in si joogto ah loola socdaa waxqabadkiisa iyo waxtarkiisa. Tani waxaa ka mid noqon kara in la ururiyo jawaab-celinta isticmaalayaasha, la qiimeeyo heerka saxnaanta ee saadaalinta modelka, iyo in la ogaado haddii ay jiraan wax isbeddelo ah oo ku yimid xogta ama xaaladaha caafimaad ee deegaanka.

5. **Hagaajinta iyo Cusboonaysiinta Modelka**: Iyadoo ku saleysan jawaab-celinta iyo qiimeynta, waxaa laga yaabaa inay lagama maarmaan noqoto in modelka la hagaajiyo ama la cusboonaysiiyo si uu ula jaanqaado xaaladaha cusub ama xogta soo baxaysa.

6. **Ka Warqabka Arrimaha Anshaxa iyo Sirta**: Markaad la shaqeynayso xogta caafimaadka, waa muhiim in si taxaddar leh loo maareeyo arrimaha la xiriira sirta iyo anshaxa. Hubi in dhammaan isticmaalka xogta uu waafaqsan yahay sharciga iyo siyaasadaha sirta xogta.

7. **Wadashaqeyn lala yeesho Khubarada Caafimaadka**: Si joogto ah ula shaqee dhakhaatiirta iyo khubarada caafimaadka si aad u hubiso in modelkaaga uu wax ku ool u yahay baahiyaha caafimaad ee maxalliga ah iyo inuu bixinayo natiijooyin macquul ah.

Ugu dambeyntii, xusuusnow in isticmaalka modelada barashada mashiinka ee go'aamada caafimaadku ay yihiin caawimaad dheeri ah oo aan beddeli karin go'aanka iyo aqoonta xirfadeed ee dhakhaatiirta iyo khubarada caafimaadka. Modelka waa in loo arkaa sida qalab taageero ah oo ka caawiya go'aan-qaadashada, halkii laga arki lahaa mid go'aanka kama dambeysta ah sameeya.

Modelka 'Random Forest Classifier' ee loo adeegsaday xogta wadnaha wuxuu leeyahay faa'iidooyin iyo adeegsiyo kala duwan, gaar ahaan marka la eego goobaha caafimaadka. Waxaa jira dhowr siyaabood oo muhiim ah oo modelkan loo isticmaali karo:

1. **Saadaalinta Xaaladaha Caafimaad**: Modelkan waxaa loo isticmaali karaa in lagu saadaaliyo halista cudurada wadnaha ee bukaanada. Tani waxay ka caawin kartaa dhakhaatiirta inay aqoonsadaan bukaanada khatar sare ugu jira xaaladaha wadnaha, taasoo u oggolaaneysa in la qaado tallaabooyin ka hortag ah ama la bilaabo daaweyn hore.

2. **Qorshaynta Daaweynta**: Iyadoo la adeegsanayo saadaasha modelka, dhakhaatiirta waxay go'aan ka gaari karaan istiraatiijiyadaha daaweynta ugu fiican, sida in bukaanada halista sare leh loo gudbiyo baaritaanno dheeraad ah ama la siiyo talooyin gaar ah oo ku saabsan hab-nololeedka caafimaadka.

3. **Ka Hortagga Cudurada Wadnaha**: Modelku wuxuu sidoo kale gacan ka geysan karaa barnaamijyada ka hortagga, isagoo awood u siinaya hay'adaha caafimaadka inay aqoonsadaan dadka khatar sare ugu jira cudurada wadnaha si loogu sameeyo la-talin iyo in la dhiirrigeliyo isbeddelo nololeed oo caafimaad leh.

4. **Waxbarashada iyo Wacyigelinta Bukaanada**: Natiijooyinka saadaalinta waxaa loo isticmaali karaa in lagu wacyigeliyo bukaanada iyo dadweynaha guud ahaan halista cudurada wadnaha iyo muhiimadda caafimaadka wadnaha.

5. **Cilmi-baaris iyo Horumarinta Daaweynta**: Xogta iyo saadaasha modelka waxaa loo adeegsan karaa cilmi-baaris ku saabsan cudurada wadnaha, taasoo gacan ka geysanaysa fahamka sababaha iyo waxyaabaha keeni kara cudurada wadnaha.

6. **Go'aan Qaadashada Degdegga ah ee Xaaladaha Degdegga ah**: In kasta oo aanan beddeli karin go'aanka dhakhtarka, modelka waxaa loo adeegsan karaa sidii qalab taageero ah oo ka caawiya dhakhaatiirta inay si dhakhso leh u aqoonsadaan xaaladaha halista ah, gaar ahaan xaaladaha degdegga ah.

Waa muhiim in la xusuusnaado in modelkan iyo kuwa kale ee barashada mashiinka aanay marnaba beddeli karin qiimeynta iyo go'aanka dhakhaatiirta xirfadleyda ah. Isticmaalkooda waa in lagu daro taxaddar iyo fahamka xadka iyo shuruudaha isticmaalkooda. Sidoo kale, waxaa muhiim ah in la hubiyo in modelka si joogto ah loo qiimeeyo oo loo cusboonaysiiyo si loo hubiyo saxnaantiisa iyo waxtarkiisa.

Si aad u cusbooneysiiso ama uga dhigto mid casri ah modelkaaga 'Random Forest Classifier', waxaa jira dhowr hab oo kala duwan oo aad qaadi karto. Ujeeddadu waa in la hubiyo in modelku uu la jaan qaado isbeddelada xogta, isbeddelada xaaladaha caafimaad, ama inuu ka jawaabo wixii caqabado ah ee la soo gudboonaaday. Halkan waxaa ah tallaabooyinka aad qaadi karto:

1. **Ururinta iyo Ku-darista Xog Cusub**: Si joogto ah ugu dar xog cusub tababarkaaga. Xogta cusub waxay ka caawin kartaa modelka inuu barto oo uu fahmo xaalado cusub ama isbeddelo ku yimid xogta.

2. **Dib-u-Tababar Modelka**: Isticmaal xogtan cusub si aad mar kale u tababarto modelka. Tani waxay ku lug leedahay dib u tababarka modelka ee xogta cusub ama isku darka xogta cusub iyo tii hore si loo abuuro model casri ah.

3. **Hagaajinta Xuduudaha Modelka**: Tijaabi xuduudaha kala duwan ee modelka. Tani waxaa ka mid noqon kara beddelidda tirada geedaha ee `n_estimators`, qoto dheerida geedaha ee `max_depth`, iyo kuwo kale. Tijaabinta xuduudahan waxay kaa caawin kartaa inaad hesho isku-dheelitirnaan wanaagsan oo u dhexeeya waxqabadka iyo guud ahaan iswaafajinta modelka.

4. **Cross-Validation iyo Qiimeynta Joogtada ah**: Isticmaal cross-validation si aad u hubiso in modelka cusub uu si joogto ah ugu shaqeynayo xogta. Qiimee waxqabadka modelka si joogto ah adigoo adeegsanaya metrikada sida saxnaanta, F1 score, iwm.

5. **Feature Engineering**: Ku dar ama ka saar sifooyinka, ama isticmaal hababka beddelka sifooyinka si aad u hesho aragtiyo cusub oo ka caawin kara modelka inuu si fiican u fahmo xogta.

6. **Dib-u-eegis iyo Tijaabinta Joogtada ah**: Si joogto ah dib ugu eeg xogta iyo natiijooyinka si aad u hubiso in modelka uu weli ku habboon yahay xaaladahaaga gaarka ah. Tani waxay sidoo kale ka caawin kartaa in la ogaado haddii ay jiraan wax isbeddelo ah oo ku yimid xogta ama duruufaha caafimaad.

7. **Isticmaalka Modello Kale oo Beddel ah**: Haddii loo baahdo, tijaabi isticmaalka noocyo kala duwan oo ah modello si aad u aragto haddii ay jiraan kuwo waxqabadkoodu ka wanaagsan yahay modelka hadda jira.

Markaad cusboonaysiinayso modelka, waxaa muhiim ah inaad hubiso in xogta cusub ay tayo sare leedahay, inaad tixgeliso arrimaha sirta iyo anshaxa, iyo inaad si taxaddar leh u qiimeyso saameynta isbeddelada aad samaynayso. Sidoo kale, waa muhiim in la hubiyo in la raaco shuruucda iyo siyaasadaha la xiriira xogta caafimaadka iyo isticmaalka modelada barashada mashiinka.
