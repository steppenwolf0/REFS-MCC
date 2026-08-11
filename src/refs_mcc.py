import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from sklearn.feature_selection import SelectorMixin

from features import loadDatasetOriginal
from reduceData import reduceDataset

from aBioInf100 import main as reduce_features
from summaryMulti import evaluate
from sumFig import create_summary_figure

class REFS_MCC(SelectorMixin):
    def __init__(self, threads: int = 10, totalRuns: int = 10, numberOfFolds: int = 10, data: str = "../data", output: str = ".", verbose: int = 1):
        self.threads = threads
        self.totalRuns = totalRuns
        self.numberOfFolds = numberOfFolds
        self.data = data
        self.output = output
        self.verbose = verbose

        self.support_ = None
        self.ranking_ = None
        self.feature_names_in_ = None
        self.n_features_in_ = 0
        self.n_features_ = 0
        self.estimators_ = None
        self.classes_ = None

    def __set_results(self, X, idsReduced, resultsFeatures):
        self.idsReduced = idsReduced
        self.n_features_in_ = X.shape[1]
        self.n_features_ = len(idsReduced)
        self.support_ = np.array([True if i in idsReduced else False for i in range(self.n_features_in_)])

        # use resultsFeatures to rank the features, everything that is in idsReduced gets a 1, 
        # the ones in resultsFeatures starts from 2 and goes up based on the count in resultsFeatures (the more the lower the rank)
        # the rest of the features that are not in resultsFeatures get a rank of len(resultsFeatures) + 2
        resultFeaturesWithoutIdsReduced = np.array([[feature, frequency] for feature, frequency in resultsFeatures[:, 0:2] if feature not in idsReduced])
        resultFeaturesWithoutIdsReduced = resultFeaturesWithoutIdsReduced[resultFeaturesWithoutIdsReduced[:, 1].argsort()[::-1]] # order by frequency in descending order
        ranking_dict = {feature: rank for rank, feature in enumerate(resultFeaturesWithoutIdsReduced[:, 0], start=2)}
        self.ranking_ = np.array([1 if i in idsReduced else ranking_dict.get(i, len(resultsFeatures) + 2) for i in range(self.n_features_in_)])

    def _get_support_mask(self):
        if self.support_ is None:
            raise ValueError("The model has not been fitted yet. Please call the 'fit' method before accessing 'get_support'.")

        return self.support_

    # run from data folder
    def run(self):
        if not os.path.exists(self.data):
            print(f"Data folder '{self.data}' does not exists. Provide a valid one to start.")
            sys.exit(1)

        if self.output != "." and os.path.exists(self.output):
            print(f"Output folder '{self.output}' already exists. Please choose a different folder or remove the existing one.")
            sys.exit(1)

        os.makedirs(self.output, exist_ok=True)

        X, y, biomarkerNames = loadDatasetOriginal(data=self.data)
        self.feature_names_in_ = biomarkerNames
        
        results, idsReduced, resultsFeatures = reduce_features(X, y, biomarkerNames, self.threads, self.totalRuns, self.numberOfFolds, output=self.output)
        self.__set_results(X, idsReduced, resultsFeatures)

        pd.DataFrame(results).to_csv(os.path.join(self.output, "best", "sum.csv"), header=False, index=False)

        dataReduced, featuresReduced = reduceDataset(X, biomarkerNames, idsReduced)

        classifierResults = evaluate(dataReduced, y, featuresReduced, self.numberOfFolds, self.output)

        create_summary_figure(pd.DataFrame(results), pd.DataFrame(featuresReduced), self.totalRuns, output=self.output)

    def fit(self, X, y, **fit_params):
        """Fit the REFS-MCC model on the selected features.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values.

        **fit_params : dict
            - criterion: matthews_corrcoef by default
            - classifierList: list of classifiers, by default an ensemble of 8

        Returns
        -------
        self : object
            Self with the results.
        """

        if self.feature_names_in_ is not None:
            biomarkerNames = self.feature_names_in_
        else: 
            biomarkerNames = np.array([i for i in range(X.shape[1])])

        results, idsReduced, resultsFeatures = reduce_features(X, y, biomarkerNames, self.threads, self.totalRuns, self.numberOfFolds, output=self.output, verbose=self.verbose, **fit_params)

        if self.verbose:
            # create folder
            folderName = os.path.join(self.output, "best")
            if not os.path.exists(folderName) : os.makedirs(folderName)
            pd.DataFrame(results).to_csv(os.path.join(folderName, "sum.csv"), header=False, index=False)

        self.__set_results(X, idsReduced, resultsFeatures)

        return self
    
    def transform(self, X):
        """Reduce X to the selected features.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        Returns
        -------
        X_r : array of shape [n_samples, n_selected_features]
            The input samples with only the selected features.
        """

        if self.feature_names_in_ is not None:
            biomarkerNames = self.feature_names_in_
        else: 
            biomarkerNames = np.array([i for i in range(X.shape[1])])

        dataReduced, featuresReduced = reduceDataset(np.array(X), biomarkerNames, self.idsReduced)
 
        if self.verbose:
            # create folder
            folderName = os.path.join(self.output, "best")
            if not os.path.exists(folderName) : os.makedirs(folderName)
            pd.DataFrame(dataReduced).to_csv(os.path.join(folderName, "data_0.csv"), header=False, index=False)

            feature_names = np.array(featuresReduced).reshape(-1, 1)
            pd.DataFrame(feature_names).to_csv(os.path.join(folderName, "features_0.csv"), header=False, index=False)

        return dataReduced



def main():
    parser = argparse.ArgumentParser(description="REFS-MCC")
    parser.add_argument('--threads', type=int, default=10, help='Number of threads (default: 10)')
    parser.add_argument('--totalRuns', type=int, default=10, help='Total number of runs (default: 10)')
    parser.add_argument('--folds', type=int, default=10, help='Number of folds (default: 10)')
    parser.add_argument('--data', type=str, default="../data", help='Path to the data folder (default: ../data)')
    parser.add_argument('--output', type=str, default=".", help='Path to the output folder (default: .)')
    args = parser.parse_args()

    refs_mcc = REFS_MCC(args.threads, args.totalRuns, args.folds, args.data, args.output)
    return refs_mcc.run()

if __name__ == "__main__":
    sys.exit(main())