### Cerinta 8
De ce se folosesc modelele:
kNN — simplu, fara presupuneri (parametri); lent la predictie, sensibil la scara feature-urilor si la zgomot.
Decision Tree — usor de interpretat
GaussianNB — rapid, probabilistic;
MLP — flexibil, non-liniar; cere de obicei scalare si tuning; cu setari default poate merge slab fata de celelalte modele.

Decision Tree (~0.898): cel mai bun dintre cele patru pe acest split.
GaussianNB (~0.891) și kNN (~0,869): apropiate, performanta buna.
MLP (~0.572): mult mai slab

### Cerinta 9
**Note**: Folosim macro pt metrici: (average="macro"): util cand vrem sa vedem daca modelul e potrivit si pe clase rare, nu doar pe cea majoritara. Ambele tipuri de boabe contează la fel (nu vreau sa optimizez doar pentru clasa mai frecventa).
La un dezechilibru atât de mic cum avem aici pe dataset, macro și weighted vor fi apropiate, deci nu schimbă dramatic rezultatul.
#### Concluzie:
Pe experimentul nostru, k mediu-mare (151–501) e cel mai potrivit pentru F1 si accuracy; k foarte mare (1523) ridica precizia dar scade recall si F1; k=51 e putin sub celelalte in seria asta.

### Cerinta 10
Rezultat: ruleaza cerinta10() pentru rezultate

### Cerinta 11
|  | predicted 0 | predicted 1 |
|--------|-------------|-------------|
| real 0 | 254         | 72          |
| real 1 | 28          | 408         |

Putem considera **clasificare binara**, deoarece in target column avem 2 clase: Cammeo si Osmancik. 


TN (True Negative)	Real negativ, prezis negativ	254
FP (False Positive)	Real negativ, prezis pozitiv (fals alarmă)	72
FN (False Negative)	Real pozitiv, prezis negativ (ratat)	28
TP (True Positive)	Real pozitiv, prezis pozitiv	408
Pe scurt:

TP / TN = modelul a ghicit bine (pozitiv/pozitiv sau negativ/negativ).
FP = a zis pozitiv, dar era negativ.
FN = a zis negativ, dar era pozitiv.
Avem:
multe predictii corecte (254 + 408)
mai multe FP decat FN (72 vs 28), deci modelul tinde sa supraestimeze clasa pozitiva (mai des zice „pozitiv” cand nu e).

### Cerinta 12
Unde sunt cele mai multe erori:
Se confunda cel mai des Cammeo (luata drept Osmancik): 72 cazuri din 326 (254+72) Cammeo reale -> 72/326 ≈ 0,221 (~22,1% erori pe clasa Cammeo).
Invers, 28 Osmancik luate drept Cammeo din 436(408+28) ->  28/436 ≈ 0,064 (~6,4%).
Eroare totala pe tot testul: 100/762 ≈ 0,131 (~13,1%).

Luam 3 exemple de boabe Real: Osmancik, Prezis: Cammeo
Ne uitam la atributele Osmancik din datele de train, ca sa intelegem de ce a fost misclassified

#### Interpretare
Au fost prezise gresit deoarece valorile atributelor aria, axis si excentricitate sunt mai apropiate de distributia clasei Cammeo decat de cea Osmancik. 