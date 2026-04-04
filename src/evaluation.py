from helper_methods import split_features_target, split_train_test
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class Evaluator:
    def __init__(self, data_set, test_size, random_state):
        self.data_set = data_set
        self.test_size = test_size
        self.random_state = random_state
        self.X, self.y = split_features_target(self.data_set)
        self.X_train, self.X_test, self.y_train, self.y_test = split_train_test(self.X, self.y, test_size, random_state)

    def accuracy(self,model):
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        return accuracy_score(self.y_test, y_pred)

    def precision(self,model):
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        return precision_score(self.y_test, y_pred, average="macro")

    def recall(self,model):
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        return recall_score(self.y_test, y_pred, average="macro")

    def f1(self,model):
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        return f1_score(self.y_test, y_pred, average="macro")

    def confusion_matrix(self,model):
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        return confusion_matrix(self.y_test, y_pred)


    def get_y(self, model):
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        return self.y_test, y_pred, self.X_train, self.y_train

