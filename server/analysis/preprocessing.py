#
# OtterTune - preprocessing.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
from abc import ABCMeta, abstractmethod

from itertools import chain, combinations, combinations_with_replacement
import numpy as np
from sklearn.preprocessing import MinMaxScaler as SklearnMinMaxScaler

from .util import is_numeric_matrix, is_lexical_matrix


# ==========================================================
#  Preprocessing Base Class
# ==========================================================
class Preprocess(object, metaclass=ABCMeta):

    @abstractmethod
    def fit(self, matrix):
        pass

    @abstractmethod
    def transform(self, matrix, copy=True):
        pass

    def fit_transform(self, matrix, copy=True):
        self.fit(matrix)
        return self.transform(matrix, copy=True)

    @abstractmethod
    def inverse_transform(self, matrix, copy=True):
        pass


# ==========================================================
#   Bin by Deciles
# ==========================================================
class Bin(Preprocess):

    def __init__(self, bin_start, axis=None):
        if axis is not None and \
                axis != 1 and axis != 0:
            raise NotImplementedError("Axis={} is not yet implemented".format(axis))
        self.deciles_ = None
        self.bin_start_ = bin_start
        self.axis_ = axis

    def fit(self, matrix):
        if self.axis_ is None:
            self.deciles_ = get_deciles(matrix, self.axis_)
        elif self.axis_ == 0:  # Bin columns
            self.deciles_ = []
            for col in matrix.T:
                self.deciles_.append(get_deciles(col, axis=None))
        elif self.axis_ == 1:  # Bin rows
            self.deciles_ = []
            for row in matrix:
                self.deciles_.append(get_deciles(row, axis=None))
        return self

    def transform(self, matrix, copy=True):
        assert self.deciles_ is not None
        if self.axis_ is None:
            res = bin_by_decile(matrix, self.deciles_,
                                self.bin_start_, self.axis_)
        elif self.axis_ == 0:  # Transform columns
            columns = []
            for col, decile in zip(matrix.T, self.deciles_):
                columns.append(bin_by_decile(col, decile,
                                             self.bin_start_, axis=None))
            res = np.vstack(columns).T
        elif self.axis_ == 1:  # Transform rows
            rows = []
            for row, decile in zip(matrix, self.deciles_):
                rows.append(bin_by_decile(row, decile,
                                          self.bin_start_, axis=None))
            res = np.vstack(rows)
        assert res.shape == matrix.shape
        return res

    def inverse_transform(self, matrix, copy=True):
        raise NotImplementedError("This method is not supported")


def get_deciles(matrix, axis=None):
    if axis is not None:
        raise NotImplementedError("Axis is not yet implemented")

    assert matrix.ndim > 0
    assert matrix.size > 0

    decile_range = np.arange(10, 101, 10)
    deciles = np.percentile(matrix, decile_range, axis=axis)
    deciles[-1] = np.Inf
    return deciles


def bin_by_decile(matrix, deciles, bin_start, axis=None):
    if axis is not None:
        raise NotImplementedError("Axis is not yet implemented")

    assert matrix.ndim > 0
    assert matrix.size > 0
    assert deciles is not None
    assert len(deciles) == 10

    binned_matrix = np.zeros_like(matrix)
    for i in range(10)[::-1]:
        decile = deciles[i]
        binned_matrix[matrix <= decile] = i + bin_start

    return binned_matrix


# ==========================================================
#   Shuffle Indices
# ==========================================================
class Shuffler(Preprocess):

    def __init__(self, shuffle_rows=True, shuffle_columns=False,
                 row_indices=None, column_indices=None, seed=0):
        self.shuffle_rows_ = shuffle_rows
        self.shuffle_columns_ = shuffle_columns
        self.row_indices_ = row_indices
        self.column_indices_ = column_indices
        np.random.seed(seed)
        self.fitted_ = False

    def fit(self, matrix):
        if self.shuffle_rows_ and self.row_indices_ is None:
            self.row_indices_ = get_shuffle_indices(matrix.data.shape[0])
        if self.shuffle_columns_ and self.column_indices_ is None:
            self.column_indices_ = get_shuffle_indices(matrix.data.shape[1])
        self.fitted_ = True

    def transform(self, matrix, copy=True):
        if not self.fitted_:
            raise Exception("The fit() function must be called before transform()")
        if copy:
            matrix = matrix.copy()

        if self.shuffle_rows_:
            matrix.data = matrix.data[self.row_indices_]
            matrix.rowlabels = matrix.rowlabels[self.row_indices_]
        if self.shuffle_columns_:
            matrix.data = matrix.data[:, self.column_indices_]
            matrix.columnlabels = matrix.columnlabels[self.column_indices_]
        return matrix

    def inverse_transform(self, matrix, copy=True):
        if copy:
            matrix = matrix.copy()

        if self.shuffle_rows_:
            inverse_row_indices = np.argsort(self.row_indices_)
            matrix.data = matrix.data[inverse_row_indices]
            matrix.rowlabels = matrix.rowlabels[inverse_row_indices]
        if self.shuffle_columns_:
            inverse_column_indices = np.argsort(self.column_indices_)
            matrix.data = matrix.data[:, inverse_column_indices]
            matrix.columnlabels = matrix.columnlabels[inverse_column_indices]
        return matrix


def get_shuffle_indices(size, seed=None):
    if seed is not None:
        assert isinstance(seed, int)
        np.random.seed(seed)
    if isinstance(size, int):
        return np.random.choice(size, size, replace=False)
    else:
        indices = []
        for d in size:
            indices.append(np.random.choice(d, d, replace=False))
        return indices


# ==========================================================
#   Polynomial Features
# ==========================================================
class PolynomialFeatures(Preprocess):
    """Compute the polynomial features of the input array.
    This code was copied and modified from sklearn's
    implementation.
    """

    def __init__(self, degree=2, interaction_only=False, include_bias=True):
        self.degree_ = degree
        self.interaction_only_ = interaction_only
        self.include_bias_ = include_bias
        self.n_input_features_ = None
        self.n_output_features_ = None

#     @property
#     def powers_(self):
#         combinations = self._combinations(self.n_input_features_, self.degree_,
#                                           self.interaction_only_,
#                                           self.include_bias_)
#         return np.vstack(np.bincount(c, minlength=self.n_input_features_)
#                          for c in combinations)

    @staticmethod
    def _combinations(n_features, degree, interaction_only, include_bias):
        comb = (combinations if interaction_only else combinations_with_replacement)
        start = int(not include_bias)
        return chain.from_iterable(comb(list(range(n_features)), i)
                                   for i in range(start, degree + 1))

    def fit(self, matrix):
        assert matrix.ndim == 2
        assert matrix.size > 0

        _, n_features = matrix.shape
        combos = self._combinations(n_features, self.degree_,
                                    self.interaction_only_,
                                    self.include_bias_)
        self.n_input_features_ = matrix.shape[1]
        self.n_output_features_ = sum(1 for _ in combos)
        return self

    def transform(self, matrix, copy=True):
        """Transform data to polynomial features
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to transform, row by row.
        Returns
        -------
        XP : np.ndarray shape [n_samples, NP]
            The matrix of features, where NP is the number of polynomial
            features generated from the combination of inputs.
        """
        assert matrix.ndim == 2
        assert matrix.size > 0

        n_samples, n_features = matrix.shape

        if n_features != self.n_input_features_:
            raise ValueError("X shape does not match training shape")

        is_numeric_type = is_numeric_matrix(matrix)
        is_lexical_type = is_lexical_matrix(matrix)
        if is_lexical_type:
            strs = matrix.reshape((matrix.size,))
            maxlen = max([len(s) for s in strs])
            dtype = "S{}".format(maxlen * 2 + 1)
        else:
            dtype = matrix.dtype

        # allocate output data
        poly_matrix = np.empty((n_samples, self.n_output_features_), dtype=dtype)

        combos = self._combinations(n_features, self.degree_,
                                    self.interaction_only_,
                                    self.include_bias_)
        for i, c in enumerate(combos):
            if is_numeric_type:
                poly_matrix[:, i] = matrix[:, c].prod(1)
            elif is_lexical_type:
                n_poly1_feats = n_features + int(self.include_bias_)
                if i >= n_poly1_feats:
                    x = "*".join(np.squeeze(matrix[:, c]).tolist())
                else:
                    x = "".join(np.squeeze(matrix[:, c]).tolist())
                poly_matrix[:, i] = x
            else:
                raise TypeError("Unsupported matrix type {}".format(matrix.dtype))

        return poly_matrix

    def inverse_transform(self, matrix, copy=True):
        raise NotImplementedError("This method is not supported")


# ==========================================================
#   Dummy Encoding
# ==========================================================
class DummyEncoder(Preprocess):

    def __init__(self, n_values, categorical_features, cat_columnlabels, noncat_columnlabels):
        from sklearn.preprocessing import OneHotEncoder

        if not isinstance(n_values, np.ndarray):
            n_values = np.array(n_values)
        if not isinstance(categorical_features, np.ndarray):
            categorical_features = np.array(categorical_features)
        # assert categorical_features.size > 0
        assert categorical_features.shape == n_values.shape
        for nv in n_values:
            if nv <= 2:
                raise Exception("Categorical features must have 3+ labels")

        self.n_values = n_values
        self.cat_columnlabels = cat_columnlabels
        self.noncat_columnlabels = noncat_columnlabels
        self.encoder = OneHotEncoder(
            n_values=n_values, categorical_features=categorical_features, sparse=False)
        self.new_labels = None
        self.cat_idxs_old = categorical_features

    def fit(self, matrix):
        self.encoder.fit(matrix)
        # determine new columnlabels
        # categorical variables are done in order specified by categorical_features
        new_labels = []
        for i, cat_label in enumerate(self.cat_columnlabels):
            low = self.encoder.feature_indices_[i]
            high = self.encoder.feature_indices_[i + 1]
            for j in range(low, high):
                # eg the categorical variable named cat_var with 5 possible values
                # turns into 0/1 variables named cat_var____0, ..., cat_var____4
                new_labels.append(cat_label + "____" + str(j - low))
        # according to sklearn documentation,
        # "non-categorical features are always stacked to the right of the matrix"
        # by observation, it looks like the non-categorical features' relative order is preserved
        # BUT: there is no guarantee made about that behavior!
        # We either trust OneHotEncoder to be sensible, or look for some other way
        new_labels += self.noncat_columnlabels
        self.new_labels = new_labels

    def transform(self, matrix, copy=True):
        # actually transform the matrix
        matrix_encoded = self.encoder.transform(matrix)
        return matrix_encoded

    def fit_transform(self, matrix, copy=True):
        self.fit(matrix)
        return self.transform(matrix)

    def inverse_transform(self, matrix, copy=True):
        n_values = self.n_values
        # If there are no categorical variables, no transformation happened.
        if len(n_values) == 0:
            return matrix

        # Otherwise, this is a dummy-encoded matrix. Transform it back to original form.
        n_features = matrix.shape[-1] - self.encoder.feature_indices_[-1] + len(n_values)
        noncat_start_idx = self.encoder.feature_indices_[-1]
        inverted_matrix = np.empty((matrix.shape[0], n_features))
        cat_idx = 0
        noncat_idx = 0
        for i in range(n_features):
            if i in self.cat_idxs_old:
                new_col = np.ones((matrix.shape[0],))
                start_idx = self.encoder.feature_indices_[cat_idx]
                for j in range(n_values[cat_idx]):
                    col = matrix[:, start_idx + j]
                    new_col[col == 1] = j
                cat_idx += 1
            else:
                new_col = np.array(matrix[:, noncat_start_idx + noncat_idx])
                noncat_idx += 1
            inverted_matrix[:, i] = new_col
        return inverted_matrix

    def total_dummies(self):
        return sum(self.n_values)


def consolidate_columnlabels(columnlabels):
    import re
    # use this to check if a label was created by dummy encoder
    p = re.compile(r'(.*)____\d+')

    consolidated_columnlabels = []
    cat_seen = set()  # avoid duplicate cat_labels
    for lab in columnlabels:
        m = p.match(lab)
        # m.group(1) is the original column name
        if m:
            if m.group(1) not in cat_seen:
                cat_seen.add(m.group(1))
                consolidated_columnlabels.append(m.group(1))
        else:
            # non-categorical variable
            consolidated_columnlabels.append(lab)
    return consolidated_columnlabels


def fix_scaler(scaler, encoder, params):
    p = 0.5
    mean = scaler.mean_
    var = scaler.var_
    n_values = encoder.n_values
    cat_start_idxs = encoder.xform_start_indices
    current_idx = 0
    cat_idx = 0
    for param in params:
        if param.iscategorical:
            if param.isboolean:
                nvals = 1
            else:
                assert cat_start_idxs[cat_idx] == current_idx
                nvals = n_values[cat_idx]
                cat_idx += 1
            cat_mean = nvals * p
            cat_var = cat_mean * (1 - p)
            mean[current_idx: current_idx + nvals] = cat_mean
            var[current_idx: current_idx + nvals] = cat_var
            current_idx += nvals
        else:
            current_idx += 1

    scaler.mean_ = mean
    scaler.var_ = var
    scaler.scale_ = np.sqrt(var)


def get_min_max(params, encoder=None):
    if encoder is not None:
        num_cat_feats = encoder.n_values.size
        nfeats = len(params) - num_cat_feats + np.sum(encoder.n_values)
        n_values = encoder.n_values
        cat_start_idxs = encoder.xform_start_indices
    else:
        num_cat_feats = 0
        nfeats = len(params)
        n_values = np.array([])
        cat_start_idxs = np.array([])

    mins = np.empty((nfeats,))
    maxs = np.empty((nfeats,))
    current_idx = 0
    cat_idx = 0
    for param in params:
        if param.iscategorical:
            if param.isboolean:
                nvals = 1
            else:
                assert cat_start_idxs[cat_idx] == current_idx
                nvals = n_values[cat_idx]
                cat_idx += 1
            mins[current_idx: current_idx + nvals] = 0
            maxs[current_idx: current_idx + nvals] = 1
            current_idx += nvals
        else:
            mins[current_idx] = param.true_range[0]  # valid_values[0]
            maxs[current_idx] = param.true_range[1]  # valid_values[-1]
            current_idx += 1
    return mins, maxs


# ==========================================================
#   Min-max scaler
# ==========================================================
class MinMaxScaler(Preprocess):

    def __init__(self, mins=None, maxs=None):
        self.scaler_ = SklearnMinMaxScaler()
        if mins is not None:
            assert isinstance(mins, np.ndarray)
            if mins.ndim == 1:
                mins = mins.reshape(1, -1)
            self.scaler_.partial_fit(mins)
            self.mins_ = mins
        else:
            self.mins_ = None
        if maxs is not None:
            assert isinstance(maxs, np.ndarray)
            if maxs.ndim == 1:
                maxs = maxs.reshape(1, -1)
            self.scaler_.partial_fit(maxs)
            self.maxs_ = maxs
        else:
            self.maxs_ = None
        self.fitted_ = self.mins_ is not None and self.maxs_ is not None

    def fit(self, matrix):
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        self.scaler_.partial_fit(matrix)
        self.mins_ = self.scaler_.data_min_
        self.maxs_ = self.scaler_.data_max_
        self.fitted_ = True
        return self

    def transform(self, matrix, copy=True):
        if not self.fitted_:
            raise Exception("Model not fitted!")
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        return self.scaler_.transform(matrix)

    def inverse_transform(self, matrix, copy=True):
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        return self.scaler_.inverse_transform(matrix)
