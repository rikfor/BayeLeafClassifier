import numpy as np
import scanpy as sc
import pandas as pd
from typing import Union
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from copy import deepcopy
from itertools import combinations


class DiscreteDistribution:
    def __init__(self) -> None:
        self.nbins = None
        self.tolerance = None
        self.bounds = None
        self.feature_names = None
        self.distribution_df = None
        self.distribution_dict = None

    def fit(self, data: Union[pd.DataFrame, np.ndarray], nbins = 8, tolerance=0.05, bounds=None):
        self.nbins = nbins
        self.tolerance = tolerance
        if bounds is None:
            if isinstance(data, np.ndarray):
                self.bounds = [np.min(data), np.max(data)]
            else:
                self.bounds = [data.min().min(), data.max().max()]
        self.bin_border = np.linspace(self.bounds[0], self.bounds[1], nbins + 1)
        if isinstance(data, pd.DataFrame):
            self.feature_names = data.columns

        bin_count = self.get_bin_count(data)
        distribution = bin_count / np.sum(bin_count, axis=0)
        self.distribution_df = pd.DataFrame(distribution, columns=self.feature_names, index=self.bin_border[:-1])
        self.distribution_dict = self.distribution_df.to_dict()
        return self

    def get_bin_count(self, data: Union[pd.DataFrame, np.ndarray]):
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()
        if len(data.shape) == 1:
            data = data.reshape((-1, 1))
        bin_count = np.zeros((self.bin_border.shape[0] - 1, data.shape[1]))
        for i in range(self.bin_border.shape[0] - 1):
            bin_count[i, :] = np.sum((self.bin_border[i] - self.tolerance <= data)
                                     & (data < self.bin_border[i + 1] + self.tolerance), axis=0)
        return bin_count
    
    def get_bins(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            assert np.all(X.columns == self.distribution_df.columns)
        else:
            X = pd.DataFrame(X)
        
        bins = pd.DataFrame(index=X.index, columns=X.columns)
        for col in X.columns:
            bins[col] = pd.cut(X[col], bins=self.distribution_df.index, labels=False, include_lowest=True)
            bins.loc[bins[col].isna(), col] = len(self.distribution_df.index)-1
        bins = bins.astype(int)
        return bins

    def get_likelihood(self, X, epsilon=0.01):
        bins = self.get_bins(X)
        likelihoods = pd.DataFrame(index=bins.index, columns=bins.columns)
        for col in bins.columns:
            likelihoods[col] = np.vectorize(self.distribution_dict[col].get)(bins[col])
        likelihoods += epsilon
        return np.prod(likelihoods, axis=1)


def calc_emd(dist1: Union[pd.DataFrame, np.ndarray, DiscreteDistribution], dist2: Union[pd.DataFrame, np.ndarray, DiscreteDistribution]):
    if not isinstance(dist1, DiscreteDistribution):
        dist1 = DiscreteDistribution().fit(dist1)

    if not isinstance(dist2, DiscreteDistribution):
        dist2 = DiscreteDistribution().fit(dist2)

    assert isinstance(dist1, DiscreteDistribution)
    assert isinstance(dist2, DiscreteDistribution)
    assert np.all(dist1.feature_names == dist2.feature_names)
    assert np.all(dist1.distribution_df.shape[1] == dist2.distribution_df.shape[1])

    merge_df = pd.concat((dist1.distribution_df, -dist2.distribution_df), axis=0)
    merge_df = merge_df.sort_index()
    cum_sum = np.cumsum(merge_df, axis=0).iloc[:-1]
    carry_length = np.diff(merge_df.index).reshape((-1, 1))
    total_length = merge_df.index[-1] - merge_df.index[0]
    emd = np.sum(np.abs(cum_sum) * carry_length, axis=0) / total_length
    emd = pd.Series(emd, index=dist1.feature_names)
    return emd

class NaiveClassifier:
    def __init__(self):
        self.column_names = None
        self.class_names = None
        self.class_prob_dist = {}

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        self.column_names = np.array(list(X.columns)) 
        self.class_names = list(set(y))
        self.class_prob_dist = {}

        for c in self.class_names:
            X_c = X.loc[y==c, :]
            self.class_prob_dist[c] = DiscreteDistribution().fit(X_c)

        return self

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]):
        if isinstance(X, pd.DataFrame):
            assert set(self.column_names).issubset(set(X.columns))
            X = X[self.column_names]
        else:
            assert X.shape[1] == len(self.column_names)

        proba = pd.DataFrame(columns=self.class_names)
        for c in self.class_names:
            proba[c] = self.class_prob_dist[c].get_likelihood(X)
        return proba

    def predict(self, X):
        self.class_names = np.array(self.class_names)
        return self.class_names[np.argmax(self.predict_proba(X).to_numpy(), axis=1)]
    

class MultiClassWrapper:
    def __init__(self, class_weights = None, include_naive=True, max_depth=None, n_estimators=200,
                 min_samples_leaf=40, max_features="sqrt", parameters = None) -> None: 
        self.include_naive = include_naive
        self.distributions = DiscreteDistribution()
        self.features_ = None
        self.parameters = parameters
        self.naive_classifier = GaussianNB()
        self.forest_classifier = RandomForestClassifier(n_jobs=-1, max_depth=max_depth,  n_estimators=n_estimators,
                                                        min_samples_leaf=min_samples_leaf, max_features=max_features, class_weight = class_weights, random_state=0)

    def fit(self, data: Union[pd.DataFrame, np.ndarray], labels):
        data = self.fit_preprocess(data, labels)
        self.distributions.fit(data)
        self.forest_classifier.fit(data, labels)
        return self

    def predict(self, data: pd.DataFrame, already_preprocessed = False, required_certainty = 0.5):
        if not already_preprocessed:
            data = self.predict_preprocess(data)
        prediction = self.forest_classifier.predict(data)
        
        return prediction

    def predict_proba(self, data, already_preprocessed=False):
        if not already_preprocessed:
            data = self.predict_preprocess(data)
        return self.forest_classifier.predict_proba(data)

    def fit_preprocess(self, data, labels):
        if isinstance(data, pd.DataFrame):
            self.features_ = list(data.columns)
        if self.include_naive:
            self.naive_classifier.fit(data, labels)
            naive_proba = self.naive_classifier.predict_proba(data)
            naive_proba = pd.DataFrame(naive_proba, columns=self.naive_classifier.classes_)
            naive_proba.index = data.index
            data = pd.concat((data, naive_proba), axis=1, ignore_index=True)
        return data

    def predict_preprocess(self, data):
        if self.features_ is not None:
            if isinstance(data, pd.DataFrame): 
                if not set(self.features_).issubset(data.columns):
                    features_missing = set(self.features_) - set(data.columns)
                    data[list(features_missing)] = 0 
                data = data[self.features_]
        if self.include_naive:
            data = self.augment_with_naive(data)
        return data

    def augment_with_naive(self, data):
        naive_proba = self.naive_classifier.predict_proba(data)
        naive_proba = pd.DataFrame(naive_proba, columns=self.naive_classifier.classes_, index=data.index)
        aug_data = pd.concat((data, naive_proba), axis=1, ignore_index=True)
        return aug_data


class OneVsOneWrapper:
    def __init__(self, classifier):
        self.classifier = classifier
        self.classifiers = {}
        self.marker_dic = None
        self.specific_marker_dic = None

    def set_marker_dic(self, marker_dic):
        self.marker_dic = marker_dic

    def set_specific_marker_dic(self, specific_marker_dic):
        self.specific_marker_dic = specific_marker_dic

    def fit(self, data, labels): 
        classes = np.unique(labels)
        for class1, class2 in combinations(classes, 2):
            clf = deepcopy(self.classifier)
            mask = np.isin(labels, [class1, class2])
            if self.marker_dic is not None:
                pair_specific_markers = self.marker_dic[(class1, class2)]
                clf.fit(data.loc[mask, pair_specific_markers], labels[mask])
            else:
                clf.fit(data[mask], labels[mask])
            self.classifiers[(class1, class2)] = clf
        return self

    def get_votes(self, data: Union[pd.DataFrame, np.ndarray]):
        votes = np.zeros((data.shape[0], len(self.classifiers))).astype(np.str_)
        all_probas = {}
        if isinstance(data, pd.DataFrame):
            index = data.index
        else:
            index = range(data.shape[0])    
        class_pairs = list(self.classifiers.keys())
        votes = pd.DataFrame(votes, index=index, columns=class_pairs)

        for (class1, class2) in class_pairs:
            clf = self.classifiers[(class1, class2)]
            preds = clf.predict(data)
            all_probas[(class1, class2)] = clf.predict_proba(data)
            
            votes[(class1, class2)] = preds

        return votes, all_probas

    def predict(self, data: Union[pd.DataFrame, np.ndarray]): 
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(np.ndarray)
        
        votes, all_probas = self.get_votes(data)
        
        votes_to_pred = votes.reset_index().drop(votes.reset_index().columns[0], axis = 1)
        
        most_likely_cell_types = votes_to_pred.apply(lambda x:x.value_counts().nlargest(3), axis=1)
        top_cts_df = most_likely_cell_types.apply(lambda row: row.sort_values(ascending=False).head(3).index.tolist(), axis=1, result_type='expand')
        top_cts_df.index = data.index
        top_cts_df.columns = ['ct_pred', 'second_most_likely', 'third_most_likely']

        all_second_likelihoods = []
        all_third_likelihoods = []
        overall_likelihoods = []
        
        for i in range(len(top_cts_df)):
            first = top_cts_df.iloc[i]['ct_pred']
            second = top_cts_df.iloc[i]['second_most_likely']
            third = top_cts_df.iloc[i]['third_most_likely']
            try:
                all_second_likelihoods += [all_probas[(first, second)][i][0]]
            except KeyError:
                all_second_likelihoods += [all_probas[(second, first)][i][1]]
            
            try:
               all_third_likelihoods += [all_probas[(first, third)][i][0]]

            except KeyError:
                all_third_likelihoods += [all_probas[(third, first)][i][1]]

            temp_overall_likelihood = []
            for key in all_probas.keys():
                if first in key:
                    ind = key.index(first)
                    temp_overall_likelihood += [all_probas[key][i][ind]]

            overall_likelihoods += [np.mean(pd.Series(temp_overall_likelihood).nsmallest(3))]

        top_cts_df['likelihood_vs_second_ct'] = all_second_likelihoods
        top_cts_df['likelihood_vs_third_ct'] = all_third_likelihoods
        top_cts_df['overall_likelihood'] = overall_likelihoods
        
        return top_cts_df
    

