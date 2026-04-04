Clasificarea tipurilor de orez
Descrierea datasetului si a problemei

Datasetul analizat contine un total de 3810 instante si 8 coloane, dintre care 7 reprezinta atribute numerice, iar una reprezinta eticheta (clasa). 

Atributele sunt: 
arie (Area)
perimetru (Perimeter)
lungimea axei mari (Major_Axis_Length)
lungimea axei mici (Minor_Axis_Length)
excentricitate (Eccentricity)
aria convexa (Convex_Area)
extindere (Extent)

Variabila tinta este „Class”, care poate lua doua valori: Cammeo si Osmancik.

Problema abordata este una de clasificare binara, scopul fiind construirea unui model capabil sa diferentieze intre cele doua tipuri de orez pe baza caracteristicilor sale.

Verificarea datelor si decizii de preprocesare
Pentru organizarea mai buna a experimentelor si pentru asigurarea reproductibilitatii rezultatelor, s-a utilizat un fisier de configurare (config.py). Acesta contine parametri constanti folositi pe parcursul intregului proces de procesare si antrenare.

fisierul de configurare are mai multe avantaje importante:
centralizeaza parametrii importanti ai experimentelor
permite modificarea usoara a setarilor fara a altera logica principala
asigura consistenta intre rulari diferite ale programului

Parametri definiti
SEED (RANDOM_STATE) – valoare fixa utilizata pentru initializarea generatorului de numere aleatoare
DEFAULT_TEST_SIZE - proportia setului de test
TARGET_COLUMN - numele coloanei tinta

Un aspect esential in experimentele de machine learning este controlul randomizarii. Operatii precum impartirea datelor in train/test, initializarea ponderilor (pentru retele neuronale), selectia subseturilor (in anumite algoritmi) depind de procese aleatoare.

Prin setarea unei valori fixe pentru SEED(RANDOM_STATE) se obtin urmatoarele beneficii:
reproductibilitate: aceleasi rezultate pot fi obtinute la fiecare rulare;
comparabilitate: modelele pot fi comparate corect intre ele, deoarece folosesc aceeasi impartire a datelor;
debugging mai usor: erorile pot fi reproduse si analizate consistent.

Decodificarea etichetei

Variabila tinta avea byte encoding (format b’Cammeo’), si a fost transformata intr-un format string (“Cammeo”), necesar pentru antrenarea modelelor.
Distributia datelor

Distributia claselor arata un usor dezechilibru:

Osmancik: 2180 instante (~57.22%)
Cammeo: 1630 instante (~42.78%)

Raportul intre clase este aproximativ 1.34, ceea ce indica un dezechilibru moderat, dar nu critic. Acest aspect trebuie totusi luat in considerare in evaluarea modelelor, mai ales pentru metrici precum recall sau f1-score.

Din punct de vedere statistic, se observa diferente clare intre cele doua clase:

Cammeo are valori medii mai mari pentru Area, Perimeter si Major_Axis_Length.
Osmancik are valori mai mici in general, dar o variabilitate similara.

Aceste diferente sugereaza ca problema este bine separabila, cel putin partial, in spatiul caracteristicilor.

Analiza tipurilor de date arata ca toate variabilele de intrare sunt de tip float64, iar variabila tinta este de tip obiect (categorica). 

Integritatea datelor

Nu exista valori lipsa in dataset, ceea ce simplifica semnificativ etapa de preprocesare.

Outlieri

Detectarea outlierilor folosind metoda IQR a relevat:

Un numar foarte mic de outlieri pentru Area (4), Convex_Area (3)
Mai multi pentru Minor_Axis_Length (65, ~1.71%)
Cateva valori pentru Eccentricity (21)

Procentul total de outlieri este redus, astfel incat nu s-a considerat necesara eliminarea acestora. 
Scalarea datelor

S-au testat doua metode de scalare:
StandardScaler
MinMaxScaler

Rezultatele arata:

Accuracy cu StandardScaler: 0.909
Accuracy cu MinMaxScaler: 0.915

MinMaxScaler a oferit performanta usor mai buna, ceea ce sugereaza ca modelele sensibile la scala (precum kNN) beneficiaza de normalizare.


 Descrierea experimentelor
Impartirea datelor

S-au realizat mai multe experimente cu diferite valori pentru:

test_size: 0.2, 0.4, 0.7
seed: 0, 5, 42

Rezultatele sunt relativ stabile:

Accuracy variaza intre ~0.866 si ~0.895
F1 variaza intre ~0.860 si ~0.891

Aceasta stabilitate indica faptul ca modelul nu este extrem de sensibil la impartirea datelor.

Cross-validation si tuning

Pentru optimizarea hiperparametrilor s-a utilizat GridSearchCV:

Pentru kNN: cel mai bun n_neighbors = 51
Pentru Decision Tree: max_depth = 15, min_samples_leaf = 100

Rezultate:

kNN: scor CV = 0.885, test accuracy = 0.866
Decision Tree: scor CV = 0.929, test accuracy = 0.916

Decision Tree a beneficiat semnificativ de tuning, depasind clar celelalte modele.

Resampling

Nu s-a aplicat resampling, deoarece dezechilibrul nu este sever. Totusi, pentru imbunatatiri viitoare s-ar putea testa tehnici precum SMOTE.


Comparatii intre clasificatori

Rezultatele obtinute pentru diferite modele:

Model
Accuracy
knn
0.869
Decision Tree
0.898
Gaussian NB 
 0.891
MLP
0.572


Observatii

Decision Tree este cel mai performant model
Gaussian NB are rezultate bune, dar sub Decision Tree
kNN este decent, dar inferior
MLP performeaza slab (probabil subantrenat sau parametri nepotriviti)

Analiza kNN in functie de k

k
Accuracy
Precision
Recall
F1
15
0.867
0.874
0.856
0.862
51
0.866
0.875
0.854
0.860
151
0.869
0.875
0.857
0.863
501
0.869
0.875
0.857
0.863
1523
0.867
0.880
0.853
0.861


Performanta este relativ stabila, ceea ce indica faptul ca problema nu este extrem de sensibila la alegerea lui k.

Interpretarea metricilor si analiza erorilor

Pentru unul dintre modele:
Accuracy: 0.869
Precision: 0.875
Recall: 0.857
F1: 0.863

Aceste valori indica un echilibru bun intre precizie si acoperire.
Matricea de confuzie:

254
72
28
408


Interpretare:
254 Cammeo corect clasificate
72 Cammeo clasificate gresit ca Osmancik
28 Osmancik clasificate gresit ca Cammeo
408 Osmancik corect clasificate

Modelul face mai multe erori pentru clasa Cammeo, ceea ce poate fi explicat prin:
suprapunerea caracteristicilor
dezechilibrul usor al claselor
Analiza erorilor
Exemplele gresite arata valori intermediare:

Area ~13600–13800
Perimeter ~467
Eccentricity ~0.87–0.88

Aceste valori sunt situate intre mediile celor doua clase, ceea ce confirma ca erorile apar in zonele de suprapunere.

Concluzii si limitari
Concluzii

Datasetul este bine structurat si fara valori lipsa
Exista o separare rezonabila intre clase
Decision Tree este cel mai performant model (accuracy ~0.916 dupa tuning)
Scalarea imbunatateste performanta, in special pentru kNN
Modelele sunt stabile fata de diferite splituri ale datelor

Limitari
Dezechilibrul claselor poate influenta usor performanta
Outlierii nu au fost tratati explicit
MLP nu a fost optimizat corespunzator
Nu s-au folosit tehnici avansate de feature engineering

Pasi urmatori propusi
Simularea unui dezechilibru si aplicarea unor metode de resampling (SMOTE, undersampling)
Optimizarea mai avansata a MLP (numar de straturi, neuroni, rata de invatare)
Testarea altor algoritmi
Analiza importantei caracteristicilor pentru clasificarea bobului de orez
Validare mai robusta folosind k-fold cross-validation


