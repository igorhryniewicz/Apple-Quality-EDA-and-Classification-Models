from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import learning_curve
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import pandas as pd

class CustomClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, preprocessing_pipeline, model, k_count=10):
        self.preprocessing_pipeline = preprocessing_pipeline
        self.model = model
        self.k_count = k_count

        self.accuracy = None
        self.f1 = None
        self.report = {}
        self.pipeline = None
        # self.__train_sizes__ = []
        # self.__train_accuracies__ = []
        # self.__train_f1_scores__ = []
        self.__test_accuracies__ = []
        self.__test_f1_scores__ = []

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        """
        Combines preprocessing pipeline with a classifier, 
        evaluates, 
        then fits the classifier to the training data.
        """

        # self.pipeline = make_pipeline(
        #     self.preprocessing_pipeline, self.model
        #     )
        
        # Make evaluation
        self.report, self.accuracy, self.f1 = self.evaluate(X=X, y=y)

        return self
    
    def predict(self, X: pd.DataFrame):
        """
        Uses fitted pipeline to predicts labels for X 
        """

        # Raise an error if the pipeline is not fitted yet
        if self.pipeline is None:
            raise ValueError("Call fit(X, y) before using predict(X) method.")
        
        return self.pipeline.predict(X=X)
    
    def evaluate(self, X: pd.DataFrame, y:pd.DataFrame):
        """
        Performs stratified KFold cross-validation, then works out `accuracy` and `f1_score`
        SKF ensures that each fold maintains the class distribution the original data has
        """

        skf = StratifiedKFold(n_splits=self.k_count, shuffle=True)

        # Initialize an empty list to store scores
        all_reports = []

        for train_index, test_index in skf.split(X, y):

            # Split the data
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Create a new pipeline for each fold to prevent data leakage
            self.pipeline = make_pipeline(
                self.preprocessing_pipeline, self.model
            )

            # Fit the data selected in the current fold to the pipeline
            self.pipeline.fit(X=X_train, y=y_train)

            # Compute the scores
            test_accuracy, test_f1 = self.score(X_test=X_test, y_test=y_test)
            self.__test_accuracies__.append(test_accuracy)
            self.__test_f1_scores__.append(test_f1)

            # Get the full evaluation report
            y_pred = self.pipeline.predict(X_test)
            report = classification_report(y_true=y_test, y_pred=y_pred, output_dict=True)
            all_reports.append(report)

            # For learning curve purposes
            # self.__train_sizes__.append(len(X_train))
            # train_accuracy, train_f1 = self.score(X_train, y_train)
            # self.__train_accuracies__.append(train_accuracy)
            # self.__train_f1_scores__.append(train_f1)
            

        # Take the average across all folds
        average_accuacy = sum(self.__test_accuracies__) / self.k_count
        average_f1_score = sum(self.__test_f1_scores__) / self.k_count

        # Summarize report across all folds
        summarized_report = self.__average_metrics(all_reports, self.k_count)

        return (pd.DataFrame(summarized_report), 
                average_accuacy, 
                average_f1_score)
    
    def score(self, X_test, y_test):
        # Raise an error if the pipeline is not fitted yet
        if self.pipeline is None:
            raise ValueError("Call fit(X, y) before using predict(X) method.")
         
        # Make predictions
        y_pred = self.pipeline.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="micro")

        return accuracy, f1
    
    def __average_metrics(self, data, num_of_splits):
        """
        This helper function takes a list of dictionaries and returns 
        a dictionary with averaged scores.

        Args:
            data: A list of dictionaries, where each dictionary 
            represents metrics for a class.

        Returns:
            A dictionary with averaged scores across all dictionaries 
            in the list.
        """
        averaged_data = {}

        for key, value in data[0].items():
            # Check if the key is a nested dictionary
            if isinstance(value, dict):
                averaged_data[key] = {subkey: sum(d[key][subkey] for d in data) / num_of_splits for subkey in value}
            
            # Single value key (accuracy)
            else:
                averaged_data[key] = sum(d[key] for d in data) / num_of_splits
        return averaged_data
    


        
    




