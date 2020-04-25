import numpy as np
from scipy._lib import doccer

# Imports of constants from _multivariate script
from scipy.stats._multivariate import _LOG_2, _doc_random_state, \
    multi_rv_generic, _PSD, _squeeze_output, multi_rv_frozen


# ==================== START OF THE LOGIC FOR MV LAPLACE =====================

_mvl_doc_default_callparams = """\
mean : array_like, optional
    Mean of the distribution (default zero)
cov : array_like, optional
    Covariance matrix of the distribution (default one)
allow_singular : bool, optional
    Whether to allow a singular covariance matrix.  (Default: False)
"""

_mvl_doc_callparams_note = \
    """Setting the parameter `mean` to `None` is equivalent to having `mean`
    be the zero-vector. The parameter `cov` can be a scalar, in which case
    the covariance matrix is the identity times that value, a vector of
    diagonal entries for the covariance matrix, or a two-dimensional
    array_like.
    """

_mvl_doc_frozen_callparams = ""

_mvl_doc_frozen_callparams_note = \
    """See class definition for a detailed description of parameters."""

mvl_docdict_params = {
    '_mvl_doc_default_callparams': _mvl_doc_default_callparams,
    '_mvl_doc_callparams_note': _mvl_doc_callparams_note,
    '_doc_random_state': _doc_random_state
}

mvl_docdict_noparams = {
    '_mvl_doc_default_callparams': _mvl_doc_frozen_callparams,
    '_mvl_doc_callparams_note': _mvl_doc_frozen_callparams_note,
    '_doc_random_state': _doc_random_state
}


class multivariate_laplace_gen(multi_rv_generic):
    r"""
    A multivariate laplace random variable.

    The `mean` keyword specifies the mean. The `cov` keyword specifies the
    covariance matrix.

    Methods
    -------
    ``pdf(x, mean=None, cov=1, allow_singular=False)``
        Probability density function.
    ``logpdf(x, mean=None, cov=1, allow_singular=False)``
        Log of the probability density function.
    ``cdf(x, mean=None, cov=1, allow_singular=False, maxpts=1000000*dim, abseps=1e-5, releps=1e-5)``
        Cumulative distribution function.
    ``logcdf(x, mean=None, cov=1, allow_singular=False, maxpts=1000000*dim, abseps=1e-5, releps=1e-5)``
        Log of the cumulative distribution function.
    ``rvs(mean=None, cov=1, size=1, random_state=None)``
        Draw random samples from a multivariate laplace distribution.
    ``entropy()``
        Compute the differential entropy of the multivariate laplace.

    Parameters
    ----------
    x : array_like
        Quantiles, with the last axis of `x` denoting the components.
    %(_mvl_doc_default_callparams)s
    %(_doc_random_state)s

    Alternatively, the object may be called (as a function) to fix the mean
    and covariance parameters, returning a "frozen" multivariate laplace
    random variable:

    rv = multivariate_laplace(mean=None, cov=1, allow_singular=False)
        - Frozen object with the same methods but holding the given
          mean and covariance fixed.

    Notes
    -----
    %(_mvl_doc_callparams_note)s

    The covariance matrix `cov` must be a (symmetric) positive
    semi-definite matrix. The determinant and inverse of `cov` are computed
    as the pseudo-determinant and pseudo-inverse, respectively, so
    that `cov` does not need to have full rank.

    The probability density function for `multivariate_laplace` is

    .. math::

        f(x) = \frac{1}{\sqrt{(2 \pi)^k \det \Sigma}}
               \exp\left( -\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu) \right),

    where :math:`\mu` is the mean, :math:`\Sigma` the covariance matrix,
    and :math:`k` is the dimension of the space where :math:`x` takes values.

    .. versionadded:: 0.14.0

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from scipy.stats import multivariate_laplace

    >>> x = np.linspace(0, 5, 10, endpoint=False)
    >>> y = multivariate_laplace.pdf(x, mean=2.5, cov=0.5); y
    array([ 0.00108914,  0.01033349,  0.05946514,  0.20755375,  0.43939129,
            0.56418958,  0.43939129,  0.20755375,  0.05946514,  0.01033349])
    >>> fig1 = plt.figure()
    >>> ax = fig1.add_subplot(111)
    >>> ax.plot(x, y)

    The input quantiles can be any shape of array, as long as the last
    axis labels the components.  This allows us for instance to
    display the frozen pdf for a non-isotropic random variable in 2D as
    follows:

    >>> x, y = np.mgrid[-1:1:.01, -1:1:.01]
    >>> pos = np.dstack((x, y))
    >>> rv = multivariate_laplace([0.5, -0.2], [[2.0, 0.3], [0.3, 0.5]])
    >>> fig2 = plt.figure()
    >>> ax2 = fig2.add_subplot(111)
    >>> ax2.contourf(x, y, rv.pdf(pos))

    """

    def __init__(self, seed=None):
        super(multivariate_laplace_gen, self).__init__(seed)
        self.__doc__ = doccer.docformat(self.__doc__, mvl_docdict_params)

    def __call__(self, mean=None, cov=1, allow_singular=False, seed=None):
        """
        Create a frozen multivariate laplace distribution.

        See `multivariate_laplace_frozen` for more information.

        """
        return multivariate_laplace_frozen(mean, cov,
                                           allow_singular=allow_singular,
                                           seed=seed)

    def _process_parameters(self, dim, mean, cov):
        """
        Infer dimensionality from mean or covariance matrix, ensure that
        mean and covariance are full vector resp. matrix.

        """

        # Try to infer dimensionality
        if dim is None:
            if mean is None:
                if cov is None:
                    dim = 1
                else:
                    cov = np.asarray(cov, dtype=float)
                    if cov.ndim < 2:
                        dim = 1
                    else:
                        dim = cov.shape[0]
            else:
                mean = np.asarray(mean, dtype=float)
                dim = mean.size
        else:
            if not np.isscalar(dim):
                raise ValueError("Dimension of random variable must be "
                                 "a scalar.")

        # Check input sizes and return full arrays for mean and cov if
        # necessary
        if mean is None:
            mean = np.zeros(dim)
        mean = np.asarray(mean, dtype=float)

        if cov is None:
            cov = 1.0
        cov = np.asarray(cov, dtype=float)

        if dim == 1:
            mean.shape = (1,)
            cov.shape = (1, 1)

        if mean.ndim != 1 or mean.shape[0] != dim:
            raise ValueError("Array 'mean' must be a vector of length %d." %
                             dim)
        if cov.ndim == 0:
            cov = cov * np.eye(dim)
        elif cov.ndim == 1:
            cov = np.diag(cov)
        elif cov.ndim == 2 and cov.shape != (dim, dim):
            rows, cols = cov.shape
            if rows != cols:
                msg = ("Array 'cov' must be square if it is two dimensional,"
                       " but cov.shape = %s." % str(cov.shape))
            else:
                msg = ("Dimension mismatch: array 'cov' is of shape %s,"
                       " but 'mean' is a vector of length %d.")
                msg = msg % (str(cov.shape), len(mean))
            raise ValueError(msg)
        elif cov.ndim > 2:
            raise ValueError("Array 'cov' must be at most two-dimensional,"
                             " but cov.ndim = %d" % cov.ndim)

        return dim, mean, cov

    def _process_quantiles(self, x, dim):
        """
        Adjust quantiles array so that last axis labels the components of
        each data point.

        """
        x = np.asarray(x, dtype=float)

        if x.ndim == 0:
            x = x[np.newaxis]
        elif x.ndim == 1:
            if dim == 1:
                x = x[:, np.newaxis]
            else:
                x = x[np.newaxis, :]

        return x

    def _logpdf(self, x, mean, prec_U, log_det_cov, rank):
        """
        Parameters
        ----------
        x : ndarray
            Points at which to evaluate the log of the probability
            density function
        mean : ndarray
            Mean of the distribution
        prec_U : ndarray
            A decomposition such that np.dot(prec_U, prec_U.T)
            is the precision matrix, i.e. inverse of the covariance matrix.
        log_det_cov : float
            Logarithm of the determinant of the covariance matrix
        rank : int
            Rank of the covariance matrix.

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'logpdf' instead.

        """
        dev = x - mean
        # maha = np.sum(np.square(np.dot(dev, prec_U)), axis=-1)
        maha = np.sum(np.abs(np.dot(dev, prec_U)), axis=-1)

        # return -0.5 * (rank * _LOG_2PI + log_det_cov + maha)
        return - (rank * _LOG_2 + 0.5 * log_det_cov + maha)

    def logpdf(self, x, mean=None, cov=1, allow_singular=False):
        """
        Log of the multivariate laplace probability density function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
        %(_mvl_doc_default_callparams)s

        Returns
        -------
        pdf : ndarray or scalar
            Log of the probability density function evaluated at `x`

        Notes
        -----
        %(_mvl_doc_callparams_note)s

        """
        dim, mean, cov = self._process_parameters(None, mean, cov)
        x = self._process_quantiles(x, dim)
        psd = _PSD(cov, allow_singular=allow_singular)
        out = self._logpdf(x, mean, psd.U, psd.log_pdet, psd.rank)
        return _squeeze_output(out)

    def pdf(self, x, mean=None, cov=1, allow_singular=False):
        """
        Multivariate laplace probability density function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
        %(_mvl_doc_default_callparams)s

        Returns
        -------
        pdf : ndarray or scalar
            Probability density function evaluated at `x`

        Notes
        -----
        %(_mvl_doc_callparams_note)s

        """
        dim, mean, cov = self._process_parameters(None, mean, cov)
        x = self._process_quantiles(x, dim)
        psd = _PSD(cov, allow_singular=allow_singular)
        out = np.exp(self._logpdf(x, mean, psd.U, psd.log_pdet, psd.rank))
        return _squeeze_output(out)

    def _cdf(self, x, mean, prec_U):
        """
        Parameters
        ----------
        x : ndarray
            Points at which to evaluate the log of the probability
            density function
        mean : ndarray
            Mean of the distribution
        prec_U : ndarray
            A decomposition such that np.dot(prec_U, prec_U.T)
            is the precision matrix, i.e. inverse of the covariance matrix.

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'logcdf' instead.

        """
        # Dev here is the nominator of expression, equals to: x - mu
        dev = x - mean
        # Equals to: (x - mu) / b
        exp_body = np.sum((np.dot(dev, prec_U)), axis=-1)
        mask_minus = exp_body < 0
        exp_body[mask_minus] *= -1  # equals to np.abs(exp_body) but faster
        # Equals to the CDF for both positive and negative branches
        cdf_val = 0.5 * np.exp(-exp_body)
        # Special analytical case when x - mu < 0
        cdf_val[mask_minus] *= -1
        cdf_val[mask_minus] += 1

        return cdf_val

    def logcdf(self, x, mean=None, cov=1, allow_singular=False):
        """
        Log of the multivariate laplace cumulative distribution function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
        %(_mvl_doc_default_callparams)s

        Returns
        -------
        cdf : ndarray or scalar
            Log of the cumulative distribution function evaluated at `x`

        Notes
        -----
        %(_mvl_doc_callparams_note)s

        .. versionadded:: 1.0.0

        """
        dim, mean, cov = self._process_parameters(None, mean, cov)
        x = self._process_quantiles(x, dim)
        psd = _PSD(cov, allow_singular=allow_singular)
        out = np.log(self._cdf(x, mean, psd.U))
        return _squeeze_output(out)

    def cdf(self, x, mean=None, cov=1, allow_singular=False):
        """
        Multivariate laplace cumulative distribution function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
        %(_mvl_doc_default_callparams)s

        Returns
        -------
        cdf : ndarray or scalar
            Cumulative distribution function evaluated at `x`

        Notes
        -----
        %(_mvl_doc_callparams_note)s

        .. versionadded:: 1.0.0

        """
        dim, mean, cov = self._process_parameters(None, mean, cov)
        x = self._process_quantiles(x, dim)
        psd = _PSD(cov, allow_singular=allow_singular)
        out = self._cdf(x, mean, psd.U)
        return _squeeze_output(out)

    def rvs(self, mean=None, cov=1, size=1, random_state=None):
        """
        Draw random samples from a multivariate laplace distribution.

        Parameters
        ----------
        %(_mvl_doc_default_callparams)s
        size : integer, optional
            Number of samples to draw (default 1).
        %(_doc_random_state)s

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of size (`size`, `N`), where `N` is the
            dimension of the random variable.

        Notes
        -----
        %(_mvl_doc_callparams_note)s

        """
        dim, mean, cov = self._process_parameters(None, mean, cov)

        random_state = self._get_random_state(random_state)
        out = random_state.multivariate_laplace(mean, cov, size)
        return _squeeze_output(out)

    def entropy(self, mean=None, cov=1):
        """
        Compute the differential entropy of the multivariate laplace.

        Parameters
        ----------
        %(_mvl_doc_default_callparams)s

        Returns
        -------
        h : scalar
            Entropy of the multivariate laplace distribution

        Notes
        -----
        %(_mvl_doc_callparams_note)s

        """
        dim, mean, cov = self._process_parameters(None, mean, cov)
        _, logdet = np.linalg.slogdet(2 * np.pi * np.e * cov)
        return 0.5 * logdet


multivariate_laplace = multivariate_laplace_gen()


class multivariate_laplace_frozen(multi_rv_frozen):
    def __init__(self, mean=None, cov=1, allow_singular=False, seed=None,
                 maxpts=None, abseps=1e-5, releps=1e-5):
        """
        Create a frozen multivariate laplace distribution.

        Parameters
        ----------
        mean : array_like, optional
            Mean of the distribution (default zero)
        cov : array_like, optional
            Covariance matrix of the distribution (default one)
        allow_singular : bool, optional
            If this flag is True then tolerate a singular
            covariance matrix (default False).
        seed : {None, int, `~np.random.RandomState`, `~np.random.Generator`}, optional
            This parameter defines the object to use for drawing random
            variates.
            If `seed` is `None` the `~np.random.RandomState` singleton is used.
            If `seed` is an int, a new ``RandomState`` instance is used, seeded
            with seed.
            If `seed` is already a ``RandomState`` or ``Generator`` instance,
            then that object is used.
            Default is None.
        maxpts: integer, optional
            The maximum number of points to use for integration of the
            cumulative distribution function (default `1000000*dim`)
        abseps: float, optional
            Absolute error tolerance for the cumulative distribution function
            (default 1e-5)
        releps: float, optional
            Relative error tolerance for the cumulative distribution function
            (default 1e-5)

        Examples
        --------
        When called with the default parameters, this will create a 1D random
        variable with mean 0 and covariance 1:

        >>> from scipy.stats import multivariate_laplace
        >>> r = multivariate_laplace()
        >>> r.mean
        array([ 0.])
        >>> r.cov
        array([[1.]])

        """
        self._dist = multivariate_laplace_gen(seed)
        self.dim, self.mean, self.cov = self._dist._process_parameters(
                                                            None, mean, cov)
        self.cov_info = _PSD(self.cov, allow_singular=allow_singular)
        if not maxpts:
            maxpts = 1000000 * self.dim
        self.maxpts = maxpts
        self.abseps = abseps
        self.releps = releps

    def logpdf(self, x):
        x = self._dist._process_quantiles(x, self.dim)
        out = self._dist._logpdf(x, self.mean, self.cov_info.U,
                                 self.cov_info.log_pdet, self.cov_info.rank)
        return _squeeze_output(out)

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def logcdf(self, x):
        return np.log(self.cdf(x))

    def cdf(self, x):
        x = self._dist._process_quantiles(x, self.dim)
        out = self._dist._cdf(x, self.mean, self.cov_info.U)
        return _squeeze_output(out)
        return self._dist.cdf(x)
        x = self._dist._process_quantiles(x, self.dim)
        out = self._dist._cdf(x, self.mean, self.cov, self.maxpts, self.abseps,
                              self.releps)
        return _squeeze_output(out)

    def rvs(self, size=1, random_state=None):
        return self._dist.rvs(self.mean, self.cov, size, random_state)

    def entropy(self):
        """
        Computes the differential entropy of the multivariate laplace.

        Returns
        -------
        h : scalar
            Entropy of the multivariate laplace distribution

        """
        log_pdet = self.cov_info.log_pdet  # log(sigma^2) = 2*log(b)
        rank = self.cov_info.rank
        return rank * (_LOG_2 + 1) + 0.5 * log_pdet  # = log(2*b*e)


# Set frozen generator docstrings from corresponding docstrings in
# multivariate_laplace_gen and fill in default strings in class docstrings
for name in ['logpdf', 'pdf', 'logcdf', 'cdf', 'rvs']:
    method = multivariate_laplace_gen.__dict__[name]
    method_frozen = multivariate_laplace_frozen.__dict__[name]
    method_frozen.__doc__ = doccer.docformat(method.__doc__,
                                             mvl_docdict_noparams)
    method.__doc__ = doccer.docformat(method.__doc__, mvl_docdict_params)

# =============================================================================
