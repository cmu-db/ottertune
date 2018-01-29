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
class Preprocess(object):

    __metaclass__ = ABCMeta

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
        return chain.from_iterable(comb(range(n_features), i)
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

    def __init__(self, n_values, feature_indices):
        import warnings
        from sklearn.preprocessing import OneHotEncoder

        if not isinstance(n_values, np.ndarray):
            n_values = np.array(n_values)
        if not isinstance(feature_indices, np.ndarray):
            feature_indices = np.array(feature_indices)
        assert feature_indices.size > 0
        assert feature_indices.shape == n_values.shape
        for nv in n_values:
            if nv <= 2:
                raise Exception("Categorical features must have 3+ labels")

        self.feature_indices = feature_indices
        self.n_values = n_values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.encoder = OneHotEncoder(n_values=n_values, sparse=False)
        self.columnlabels = None
        self.xform_start_indices = None

    def fit(self, matrix):
        return self.fit_with_columnlabels(matrix, columnlabels=None)

    def fit_with_columnlabels(self, matrix, columnlabels=None):
        assert isinstance(matrix, np.ndarray)
        X_cat = matrix[:, self.feature_indices]
        self.encoder.fit(X_cat)

        self.xform_start_indices = np.empty_like(self.feature_indices)
        for i, (idx, nvals) in enumerate(zip(self.feature_indices, self.n_values)):
            start_idx = idx + np.sum(self.n_values[:i]) - np.sum(self.n_values[:i].size)
            self.xform_start_indices[i] = start_idx

        if columnlabels is not None:
            labels = []
            cat_index = 0
            for i in range(matrix.shape[1]):
                orig_label = columnlabels[i]
                if i in self.feature_indices:
                    assert self.feature_indices[cat_index] == i
                    nvals = self.n_values[cat_index]
                    labels.extend(["{}#{}".format(orig_label, v)
                                   for v in range(nvals)])
                    cat_index += 1
                else:
                    labels.append(orig_label)
            self.columnlabels = np.array(labels)
        return self

    def transform(self, matrix, copy=True):
        num_cat_feats = self.feature_indices.size
        X_cat = matrix[:, self.feature_indices]
        X_enc = self.encoder.transform(X_cat)
        assert X_enc.shape[1] == np.sum(self.n_values)

        nfeats = matrix.shape[1] - num_cat_feats + np.sum(self.n_values)
        offset = 0
        cat_index = 0
        new_matrix = []
        for i in range(matrix.shape[1]):
            if i in self.feature_indices:
                assert self.feature_indices[cat_index] == i
                nvals = self.n_values[cat_index]
                new_matrix.append(X_enc[:, offset: offset + nvals])
                offset += nvals
                cat_index += 1
            else:
                new_matrix.append(matrix[:, i].reshape(matrix.shape[0], 1))

        new_matrix = np.hstack(new_matrix)
        assert new_matrix.shape == (matrix.shape[0], nfeats)
        return new_matrix

    def inverse_transform(self, matrix, copy=True):
        assert matrix.ndim == 2
        n_cat_feats = self.n_values.size
        cat_idx = 0
        current_idx = 0
        nsamples = matrix.shape[0]
        nfeats = matrix.shape[1] - np.sum(self.n_values) + n_cat_feats
        new_matrix = np.empty((nsamples, nfeats))
        for i in range(nfeats):
            if cat_idx < n_cat_feats and current_idx == self.xform_start_indices[cat_idx]:
                new_col = np.ones((nsamples,)) * np.nan
                nvals = self.n_values[cat_idx]
                for n in range(nvals):
                    col = matrix[:, current_idx + n]
                    new_col[col == 1] = n
                assert np.all(np.isfinite(new_col))
                current_idx += nvals
                cat_idx += 1
            else:
                new_col = np.array(matrix[:, current_idx])
                current_idx += 1
            new_matrix[:, i] = new_col
        return new_matrix


def dummy_encoder_helper(config_mgr, featured_knobs):
    # Note: this function will not work without a config manager.
    # It just needs the type information about each of the knobs
    # being passed in.
    cat_knob_indices = []
    n_values = []
    params = []
    for i, knob_name in enumerate(featured_knobs):
        knob = config_mgr._find_param(knob_name)  # pylint: disable=protected-access
        params.append(knob)
        if knob.iscategorical and knob.data_type != "boolean":
            cat_knob_indices.append(i)
            n_values.append(len(knob.valid_values))
    cat_knob_indices = np.array(cat_knob_indices)
    n_values = np.array(n_values)
    return n_values, cat_knob_indices, params


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


# ==========================================================
#   Testing
# ==========================================================

# def test_preprocess_module():
#     import warnings
#     from .util import arrays_share_data
#
#     warnings.filterwarnings('error')
#
#     assert issubclass(Bin, Preprocess)
#     assert isinstance(Bin(bin_start=1), Preprocess)
#     assert issubclass(PolynomialFeatures, Preprocess)
#     assert isinstance(PolynomialFeatures(), Preprocess)
#
#     x1 = np.array([[2, 7, 9],
#                    [6, 9, 2],
#                    [4, 0, 2],
#                    [7, 2, 5]], dtype="float64")
#
#     print "Testing 'Bin'..."
#     decile_range = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
#     deciles_exp = np.percentile(x1, decile_range)
#     deciles_exp[-1] = np.Inf
#     bin_pp = Bin(bin_start=1)
#     x1_binned = bin_pp.fit_transform(x1)
#     assert np.array_equal(deciles_exp, bin_pp.deciles_)
#     assert is_numeric_matrix(x1)
#     assert not is_lexical_matrix(x1)
#
#     x1_binned_exp = np.array([[1, 8, 10],
#                               [7, 10, 1],
#                               [5, 1, 1],
#                               [8, 1, 6]], dtype="float64")
#     assert np.array_equal(x1_binned_exp, x1_binned)
#     assert not arrays_share_data(x1, x1_binned)
#
#     # Test transform with floats out of original range
#     x2 = np.array([-1., 6., 20.])
#     x2_binned_exp = np.array([1, 7, 10])
#     x2_binned = bin_pp.transform(x2)
#     assert np.array_equal(x2_binned_exp, x2_binned)
#
#     # Test empty array
#     x_empty = np.array([], dtype="float64")
#     bin_pp = Bin(0)
#     try:
#         bin_pp.fit_transform(x_empty)
#         print "Bin: failed empty array test"
#     except AssertionError:
#         print "Bin: passed empty array test"
#
#     print "Passed all tests for 'Bin'"
#     print ""
#     print "Testing 'PolynomialFeatures'..."
#
#     x1_poly_exp = np.array([[1, 8, 10, 1, 8, 10, 64, 80, 100],
#                             [7, 10, 1, 49, 70, 7, 100, 10, 1],
#                             [5, 1, 1, 25, 5, 5, 1, 1, 1],
#                             [8, 1, 6, 64, 8, 48, 1, 6, 36]], dtype="float64")
#
#     x1_poly_inter_exp = np.array([[1, 8, 10, 8, 10, 80],
#                                   [7, 10, 1, 70, 7, 10],
#                                   [5, 1, 1, 5, 5, 1],
#                                   [8, 1, 6, 8, 48, 6]], dtype="float64")
#
#     poly_pp = PolynomialFeatures(include_bias=False)
#     x1_poly = poly_pp.fit_transform(x1_binned_exp)
#     assert np.array_equal(x1_poly_exp, x1_poly)
#     poly_pp = PolynomialFeatures(include_bias=True)
#     x1_poly = poly_pp.fit_transform(x1_binned_exp)
#     x1_poly_bias_exp = np.hstack([np.ones((x1_binned_exp.shape[0], 1)), x1_poly_exp])
#     assert np.array_equal(x1_poly_bias_exp, x1_poly)
#     poly_pp = PolynomialFeatures(include_bias=False, interaction_only=True)
#     x1_poly = poly_pp.fit_transform(x1_binned_exp)
#     assert np.array_equal(x1_poly_inter_exp, x1_poly)
#
#     x_alpha = np.array([['a', 'b', 'c']], dtype=object)
#     x_alpha_exp = np.array([['', 'a', 'b', 'c', 'aa', 'ab', 'ac', 'bb', 'bc', 'cc']])
#     assert not is_numeric_matrix(x_alpha)
#     assert is_lexical_matrix(x_alpha)
#     poly_pp = PolynomialFeatures()
#     x_alpha_poly = poly_pp.fit_transform(x_alpha)
#     assert np.array_equal(x_alpha_exp, x_alpha_poly)
#
#     print "Passed all tests for 'PolynomialFeatures'"
#     print ""


# if __name__ == '__main__':
#     test_preprocess_module()
