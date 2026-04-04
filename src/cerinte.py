from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
# from imblearn.over_sampling import SMOTE


from src.data_loading import print_class_distribution, FullReport
from src.evaluation import Evaluator
from config import SEED
from src.helper_methods import split_features_target, split_train_test

def cerinta1_2_3_4(dataset):
    print("Shape:", dataset.shape)
    X, y = split_features_target(dataset)
    print("Features:", list(X.columns))
    print("Classes:", sorted(y.unique().tolist()))
    print_class_distribution(y)
    print()
    FullReport(dataset).print_report()
    print()

def cerinta5(dataset):
    variante_seed = [0,5,42]
    variante_proportii = [0.2,0.4,0.7]

    for seed in variante_seed:
        for proportie in variante_proportii:
            print(f"Model pt test_size={proportie} si seed={seed}")
            evaluator = Evaluator(dataset,proportie,seed)
            model = KNeighborsClassifier(n_neighbors=51)
            accuracy_score = evaluator.accuracy(model)
            precision_score = evaluator.precision(model)
            recall_score = evaluator.recall(model)
            f1_score = evaluator.f1(model)
            print(f"Accuracy: {accuracy_score:.3f}, Precision: {precision_score:.3f}, Recall: {recall_score:.3f}, F1: {f1_score:.3f}")

def cerinta6(dataset, test_size, random_state):
    X, y = split_features_target(dataset)
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size, random_state)

    scaler_standard = StandardScaler()
    scaler_standard.fit(X_train)
    X_train_scaler_standard = scaler_standard.transform(X_train)
    X_test_scaler_standard = scaler_standard.transform(X_test)

    model_knn_saler_standard = KNeighborsClassifier(n_neighbors=51)
    model_knn_saler_standard.fit(X_train_scaler_standard, y_train)
    y_pred_scaler_standard = model_knn_saler_standard.predict(X_test_scaler_standard)

    scaler_minmax = MinMaxScaler()
    scaler_minmax.fit(X_train)
    X_train_scaler_minmax = scaler_minmax.transform(X_train)
    X_test_scaler_minmax = scaler_minmax.transform(X_test)

    model_knn_scaler_minmax = KNeighborsClassifier(n_neighbors=51)
    model_knn_scaler_minmax.fit(X_train_scaler_minmax, y_train)
    y_pred_scaler_minmax = model_knn_scaler_minmax.predict(X_test_scaler_minmax)

    print(f"Accuracy cu scalare standard:  {accuracy_score(y_test, y_pred_scaler_standard):.3f}")
    print(f"Accuracy cu scalare MinMax:  {accuracy_score(y_test, y_pred_scaler_minmax):.3f}")

#Problema de import
# def cerinta7(dataset, test_size, random_state):
#     X, y = split_features_target(dataset)
#     print_class_distribution(y[:2500])
#     X_train, X_test, y_train, y_test = split_train_test(X[:2500], y[:2500], test_size, random_state)
#     smote = SMOTE(random_state=SEED)
#     X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
#     knn_smote = KNeighborsClassifier(n_neighbors=51)
#     knn_smote.fit(X_train_smote, y_train_smote)
#     accuracy_knn_smote = accuracy_score(y_test, knn_smote.predict(X_test))
#     print(f"Accuracy pt knn smote: {accuracy_knn_smote:.3f}")
#
#
#     knn_simplu = KNeighborsClassifier(n_neighbors=51)
#     knn_simplu.fit(X_train, y_train)
#     accuracy_knn_simplu = accuracy_score(y_test, knn_simplu.predict(X_test))
#     print(f"Accuracy pt knn simplu: {accuracy_knn_simplu:.3f}")
#

def cerinta8(dataset, test_size, random_state):
    evaluator = Evaluator(dataset, test_size, random_state)

    model_knn = KNeighborsClassifier(n_neighbors=500)
    rezultat_accuracy_knn = evaluator.accuracy(model_knn)
    print(f"Pentru kNN avem accuracy {rezultat_accuracy_knn:.3f}")

    model_decision_tree = DecisionTreeClassifier(criterion="gini", max_depth=None, min_samples_leaf=1, random_state=SEED)
    rezultat_accuracy_detree = evaluator.accuracy(model_decision_tree)
    print(f"Pentru decision Tree avem accuracy {rezultat_accuracy_detree:.3f}")

    model_gaussian_nb = GaussianNB()
    rezultat_accuracy_gaussian_nb = evaluator.accuracy(model_gaussian_nb)
    print(f"Pentru Gaussian avem accuracy {rezultat_accuracy_gaussian_nb:.3f}")

    model_mlp = MLPClassifier(hidden_layer_sizes=(100,), random_state=SEED)
    rezultat_model_mlp = evaluator.accuracy(model_mlp)
    print(f"Pentru MLP avem accuracy {rezultat_model_mlp:.3f}")

def cerinta9(dataset, test_size, random_state):
    k_values = [15, 51, 151, 501, 1523]
    for k in k_values:
        model_knn = KNeighborsClassifier(n_neighbors=k)
        evaluator = Evaluator(dataset, test_size, random_state)
        rezultat_accuracy = evaluator.accuracy(model_knn)
        rezultat_precision = evaluator.precision(model_knn)
        rezultat_recall = evaluator.recall(model_knn)
        rezultat_f1 = evaluator.f1(model_knn)
        print(f"Pentru k in knn setat la {k} avem accuracy {rezultat_accuracy:.3f}")
        print(f"Pentru k in knn setat la {k} avem precizie {rezultat_precision:.3f}")
        print(f"Pentru k in knn setat la {k} avem recall {rezultat_recall:.3f}")
        print(f"Pentru k in knn setat la {k} avem f1 {rezultat_f1:.3f}")


def cerinta10(dataset, test_size, random_state):
    model_knn = KNeighborsClassifier(n_neighbors=500)
    X, y = split_features_target(dataset)
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size, random_state)
    grid_search = GridSearchCV(model_knn, param_grid={'n_neighbors': [15, 51, 151, 501, 1523]}, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    grid_search.fit(X_train, y_train)
    print(f"kNN GridSearchCV — cel mai bun scor (accuracy): {grid_search.best_score_:.3f}")
    print(f"kNN GridSearchCV — cei mai buni parametri: {grid_search.best_params_}")
    test_accuracy = grid_search.score(X_test, y_test)
    print(f"kNN GridSearchCV — accuracy pe test cu modelul ales: {test_accuracy:.3f}")

    model_decision_tree = DecisionTreeClassifier(criterion="gini", max_depth=None, random_state=SEED)
    grid_search = GridSearchCV(model_decision_tree, param_grid={'max_depth': [15, 51, 151, 501, 1523], 'min_samples_leaf': [3, 10, 100]}, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    grid_search.fit(X_train, y_train)
    print(f"Decision Tree GridSearchCV — cel mai bun scor (accuracy): {grid_search.best_score_:.3f}")
    print(f"Decision Tree GridSearchCV — cei mai buni parametri: {grid_search.best_params_}")
    test_accuracy = grid_search.score(X_test, y_test)
    print(f"Decision Tree GridSearchCV — accuracy pe test cu modelul ales: {test_accuracy:.3f}")

def cerinta11(dataset, test_size, random_state):
    model_knn = KNeighborsClassifier(n_neighbors=500)
    evaluator = Evaluator(dataset, test_size, random_state)
    rezultat_confusion_matrix = evaluator.confusion_matrix(model_knn)
    print(f"Matrice de confuzie:\n {rezultat_confusion_matrix}")
    print(model_knn.classes_)


    rezultat_accuracy = evaluator.accuracy(model_knn)
    rezultat_precision = evaluator.precision(model_knn)
    rezultat_recall = evaluator.recall(model_knn)
    rezultat_f1 = evaluator.f1(model_knn)
    print(f"accuracy {rezultat_accuracy:.3f}")
    print(f"precizie {rezultat_precision:.3f}")
    print(f"recall {rezultat_recall:.3f}")
    print(f"f1 {rezultat_f1:.3f}")

def cerinta12(dataset, test_size, random_state):
    model_knn = KNeighborsClassifier(n_neighbors=500)
    evaluator = Evaluator(dataset, test_size, random_state)

    X, y = split_features_target(dataset)
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size, random_state)
    model_knn.fit(X_train, y_train)

    y_test, y_pred, X_train, y_train = evaluator.get_y(model_knn)
    counter = 0
    for i in range(len(y_pred)):
        if((y_test.iloc[i] != y_pred[i]) & (counter<3) & (y_test.iloc[i] == "Osmancik")):
            counter = counter + 1
            print(f"Real: {y_test.iloc[i]}, Prezis: {y_pred[i]} cu index in dataset: {y_test.index[i]}")
            print(f"Atribute:\n{X_test.loc[y_test.index[i]].to_string()}")
    y_train_cammeo = y_train[y_train == "Cammeo"].index
    print(f"Atribute Cammeo:\n {X_train.loc[y_train_cammeo].describe().to_string()}")
    y_train_osmancik = y_train[y_train == "Osmancik"].index
    print(f"Atribute Osmancik:\n {X_train.loc[y_train_osmancik].describe().to_string()}")


