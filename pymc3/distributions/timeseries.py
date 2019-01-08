import theano.tensor as tt
from theano import scan

from pymc3.util import get_variable_name
from .continuous import get_tau_sigma, Normal, Flat
from . import multivariate
from . import distribution
from .distribution import (
    _DrawValuesContext,
    draw_values,
    to_tuple,
    generate_samples,
    broadcast_distribution_samples,
)
import numpy as np
from scipy import stats


__all__ = [
    'AR1',
    'AR',
    'GaussianRandomWalk',
    'GARCH11',
    'EulerMaruyama',
    'MvGaussianRandomWalk',
    'MvStudentTRandomWalk'
]


class AR1(distribution.Continuous):
    """
    Autoregressive process with 1 lag.

    Parameters
    ----------
    k : tensor
       effect of lagged value on current value
    tau_e : tensor
       precision for innovations
    """

    def __init__(self, k, tau_e, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k = tt.as_tensor_variable(k)
        self.tau_e = tau_e = tt.as_tensor_variable(tau_e)
        self.tau = tau_e * (1 - k ** 2)
        self.mode = tt.as_tensor_variable(0.)

    def logp(self, x):
        k = self.k
        tau_e = self.tau_e

        x_im1 = x[:-1]
        x_i = x[1:]
        boundary = Normal.dist(0., tau=tau_e).logp

        innov_like = Normal.dist(k * x_im1, tau=tau_e).logp(x_i)
        return boundary(x[0]) + tt.sum(innov_like)

    def _random(self, k, tau_e, sample_size=None, size=None):
        if isinstance(sample_size, np.ndarray):
            sample_size = sample_size.ravel()[0]
        if size is not None:
            self_shape = to_tuple(size)
        else:
            self_shape = to_tuple(self.shape)
        _, k, tau_e = broadcast_distribution_samples((np.empty(self_shape),
                                                      k,
                                                      tau_e),
                                                     size=sample_size)
        size = k.shape
        samples = stats.norm.rvs(loc=0, scale=1./np.sqrt(tau_e),
                                 size=size)
        for t_ind in range(1, samples.shape[-1]):
            samples[..., t_ind] += k[..., t_ind - 1] * samples[..., t_ind - 1]
        return samples

    def random(self, point=None, size=None):
        """
        Draw random values from AR1 distribution.

        Parameters
        ----------
        point : dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size : int, optional
            Desired size of random sample (returns one sample if not
            specified).

        Returns
        -------
        array
        """
        k, tau_e = broadcast_distribution_samples(draw_values([self.k,
                                                               self.tau_e],
                                                              point=point,
                                                              size=size),
                                                  size=size)
        if size is not None:
            self_shape = to_tuple(size) + to_tuple(self.shape)
        else:
            self_shape = self.shape
        broadcast_shape = broadcast_distribution_samples((np.empty(self_shape),
                                                          k),
                                                         size=size)[0].shape
        return generate_samples(self._random, k=k, tau_e=tau_e,
                                sample_size=size,
                                dist_shape=self.shape,
                                broadcast_shape=broadcast_shape,
                                size=size)

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        k = dist.k
        tau_e = dist.tau_e
        name = r'\text{%s}' % name
        return r'${} \sim \text{{AR1}}(\mathit{{k}}={},~\mathit{{tau_e}}={})$'.format(name,
                 get_variable_name(k), get_variable_name(tau_e))


class AR(distribution.Continuous):
    R"""
    Autoregressive process with p lags.

    .. math::

       x_t = \rho_0 + \rho_1 x_{t-1} + \ldots + \rho_p x_{t-p} + \epsilon_t,
       \epsilon_t \sim N(0,\sigma^2)

    The innovation can be parameterized either in terms of precision
    or standard deviation. The link between the two parametrizations is
    given by

    .. math::

       \tau = \dfrac{1}{\sigma^2}

    Parameters
    ----------
    rho : tensor
        Tensor of autoregressive coefficients. The first dimension is the p
        lag. If constant is `True`, then
        `rho = [\rho_0, \rho_1, \ldots, \rho_p]`.
        If constant is `False`, then `rho = [\rho_1, \ldots, \rho_p]`
    sigma : float
        Standard deviation of innovation (sigma > 0). (only required if tau is not specified)
    tau : float
        Precision of innovation (tau > 0). (only required if sigma is not specified)
    constant: bool (optional, default = False)
        Whether to include a constant.
    init : distribution
        distribution for initial values. The first p elements of the timeseries
        are assumed to come from the init distribution and have no AR
        operations applied to them (Defaults to Flat())
    """

    def __init__(self, rho, sigma=None, tau=None,
                 constant=False, init=Flat.dist(),
                 sd=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if sd is not None:
            sigma = sd

        tau, sigma = get_tau_sigma(tau=tau, sigma=sigma)
        self.sigma = self.sd = tt.as_tensor_variable(sigma)
        self.tau = tt.as_tensor_variable(tau)

        self.mean = tt.as_tensor_variable(0.)

        if isinstance(rho, list):
            p = len(rho)
        else:
            try:
                shape_ = rho.shape.tag.test_value
            except AttributeError:
                shape_ = rho.shape

            if hasattr(shape_, "size") and shape_.size == 0:
                p = 1
            else:
                p = shape_[0]

        if constant:
            self.p = p - 1
        else:
            self.p = p

        self.constant = constant
        if not self.constant and self.p == 1 and isinstance(rho, list):
            rho = rho[0]
        self.rho = rho = tt.as_tensor_variable(rho)
        self.init = init

    def logp(self, value):
        if self.constant:
            x = tt.add(*[self.rho[i + 1] * value[self.p - (i + 1):-(i + 1)] for i in range(self.p)])
            eps = value[self.p:] - self.rho[0] - x
        else:
            if self.p == 1:
                x = self.rho * value[:-1]
            else:
                x = tt.add(*[self.rho[i] * value[self.p - (i + 1):-(i + 1)] for i in range(self.p)])
            eps = value[self.p:] - x

        innov_like = Normal.dist(mu=0.0, tau=self.tau).logp(eps)
        init_like = self.init.logp(value[:self.p])

        return tt.sum(innov_like) + tt.sum(init_like)

    def _random(self, *args, sigma=None,
                sample_size=None, size=None):
        if self.constant:
            arg_break = self.p + 1
        else:
            arg_break = self.p
        rho = args[:arg_break]
        init = np.moveaxis(np.array(args[arg_break:]), 0, -1)
        if isinstance(sample_size, np.ndarray):
            sample_size = sample_size.ravel()[0]
        if size is not None:
            self_shape = to_tuple(size)
        else:
            self_shape = to_tuple(self.shape)
        temp = broadcast_distribution_samples(
            (np.empty(self_shape),) + rho + (sigma,),
            size=sample_size
        )
        sigma = temp[-1]
        rho = np.array(temp[1:-1])
        if self.constant:
            rho0 = rho[0]
            rho = rho[1:]
        else:
            rho0 = np.zeros_like(rho[0])
        rvs_size = sigma.shape[:-1] + (sigma.shape[-1] - self.p,)
        epsilon = stats.norm.rvs(loc=0, scale=sigma[..., :-self.p],
                                 size=rvs_size)
        samples = np.zeros(sigma.shape, dtype=self.dtype)
        samples[..., :self.p] = init
        samples[..., self.p:] = epsilon + rho0[..., :-self.p]
        for t_ind in range(self.p, samples.shape[-1]):
            for p_ind in range(self.p):
                samples[..., t_ind] += (rho[p_ind, ..., t_ind - (p_ind + 1)] *
                                        samples[..., t_ind - (p_ind + 1)])
        return samples

    def _get_rho_from_sample(self, rho, size=None):
        size = to_tuple(size)
        rho = np.atleast_1d(rho)
        p = self.p
        if self.constant:
            p += 1
        if p == 1:
            if rho.shape == size:
                rho = rho[..., np.newaxis]
                p_axis = rho.ndim - 1
            elif rho.shape[:len(size)] == size:
                if rho.ndim < len(size) or rho[len(size)] != 1:
                    rho = rho[tuple([slice(None)] * len(size), np.newaxis)]
                    p_axis = len(size)
                else:
                    p_axis = len(size)
            elif rho.shape[0] != 1:
                rho = rho[np.newaxis, :]
                p_axis = 0
            else:
                p_axis = 0
        elif rho.shape[:len(size)] == size:
            p_axis = list(rho.shape[len(size):]).index(p) + len(size)
        else:
            p_axis = list(rho.shape).index(p)
        return np.moveaxis(rho, p_axis, 0)

    def _get_init_from_sample(self, init, size=None):
        size = to_tuple(size)
        init = np.asarray(init)
        if init.shape == size:
            init = init[..., np.newaxis]
        elif init.shape[-1] not in [self.p, 1]:
            init = init[..., np.newaxis]
        return np.moveaxis(init, -1, 0)

    def random(self, point=None, size=None):
        """
        Draw random values from AR distribution.

        Parameters
        ----------
        point : dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size : int, optional
            Desired size of random sample (returns one sample if not
            specified).

        Returns
        -------
        array
        """
        with _DrawValuesContext():
            rho, sigma = draw_values([self.rho, self.sigma], point=point,
                                     size=size)
            init = self.init.random(point=point, size=size)
        size = to_tuple(size)
        rho = tuple(self._get_rho_from_sample(rho, size=size))
        init = tuple(self._get_init_from_sample(init, size=size))
        parameters = broadcast_distribution_samples(
            rho + init + (sigma,), size=size
        )
        sigma = parameters[-1]
        init = tuple(parameters[len(rho):(len(rho) + len(init))])
        rho = tuple(parameters[:len(rho)])
        if size is not None:
            self_shape = to_tuple(size) + to_tuple(self.shape)
        else:
            self_shape = self.shape
        broadcast_shape = broadcast_distribution_samples((np.empty(self_shape),
                                                          sigma),
                                                         size=size)[0].shape
        return generate_samples(self._random, *(rho + init),
                                sigma=sigma,
                                sample_size=size,
                                dist_shape=self.shape,
                                broadcast_shape=broadcast_shape,
                                size=size)


class GaussianRandomWalk(distribution.Continuous):
    R"""
    Random Walk with Normal innovations

    Parameters
    ----------
    mu: tensor
        innovation drift, defaults to 0.0
    sigma : tensor
        sigma > 0, innovation standard deviation (only required if tau is not specified)
    tau : tensor
        tau > 0, innovation precision (only required if sigma is not specified)
    init : distribution
        distribution for initial value (Defaults to Flat())
    """

    def __init__(self, tau=None, init=Flat.dist(), sigma=None, mu=0.,
                 sd=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if sd is not None:
            sigma = sd
        tau, sigma = get_tau_sigma(tau=tau, sigma=sigma)
        self.tau = tau = tt.as_tensor_variable(tau)
        self.sigma = self.sd = sigma = tt.as_tensor_variable(sigma)
        self.mu = mu = tt.as_tensor_variable(mu)
        self.init = init
        self.mean = tt.as_tensor_variable(0.)

    def logp(self, x):
        tau = self.tau
        sigma = self.sigma
        mu = self.mu
        init = self.init

        x_im1 = x[:-1]
        x_i = x[1:]

        innov_like = Normal.dist(mu=x_im1 + mu, sigma=sigma).logp(x_i)
        return init.logp(x[0]) + tt.sum(innov_like)

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        mu = dist.mu
        sigma = dist.sigma
        name = r'\text{%s}' % name
        return r'${} \sim \text{{GaussianRandomWalk}}(\mathit{{mu}}={},~\mathit{{sigma}}={})$'.format(name,
                                                get_variable_name(mu),
                                                get_variable_name(sigma))


class GARCH11(distribution.Continuous):
    R"""
    GARCH(1,1) with Normal innovations. The model is specified by

    .. math::
        y_t = \sigma_t * z_t

    .. math::
        \sigma_t^2 = \omega + \alpha_1 * y_{t-1}^2 + \beta_1 * \sigma_{t-1}^2

    with z_t iid and Normal with mean zero and unit standard deviation.

    Parameters
    ----------
    omega : tensor
        omega > 0, mean variance
    alpha_1 : tensor
        alpha_1 >= 0, autoregressive term coefficient
    beta_1 : tensor
        beta_1 >= 0, alpha_1 + beta_1 < 1, moving average term coefficient
    initial_vol : tensor
        initial_vol >= 0, initial volatility, sigma_0
    """

    def __init__(self, omega, alpha_1, beta_1,
                 initial_vol, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.omega = omega = tt.as_tensor_variable(omega)
        self.alpha_1 = alpha_1 = tt.as_tensor_variable(alpha_1)
        self.beta_1 = beta_1 = tt.as_tensor_variable(beta_1)
        self.initial_vol = tt.as_tensor_variable(initial_vol)
        self.mean = tt.as_tensor_variable(0.)

    def get_volatility(self, x):
        x = x[:-1]

        def volatility_update(x, vol, w, a, b):
            return tt.sqrt(w + a * tt.square(x) + b * tt.square(vol))

        vol, _ = scan(fn=volatility_update,
                      sequences=[x],
                      outputs_info=[self.initial_vol],
                      non_sequences=[self.omega, self.alpha_1,
                                     self.beta_1])
        return tt.concatenate([[self.initial_vol], vol])

    def logp(self, x):
        vol = self.get_volatility(x)
        return tt.sum(Normal.dist(0., sigma=vol).logp(x))

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        omega = dist.omega
        alpha_1 = dist.alpha_1
        beta_1 = dist.beta_1
        name = r'\text{%s}' % name
        return r'${} \sim \text{GARCH}(1,~1,~\mathit{{omega}}={},~\mathit{{alpha_1}}={},~\mathit{{beta_1}}={})$'.format(
            name,
            get_variable_name(omega),
            get_variable_name(alpha_1),
            get_variable_name(beta_1))


class EulerMaruyama(distribution.Continuous):
    R"""
    Stochastic differential equation discretized with the Euler-Maruyama method.

    Parameters
    ----------
    dt : float
        time step of discretization
    sde_fn : callable
        function returning the drift and diffusion coefficients of SDE
    sde_pars : tuple
        parameters of the SDE, passed as *args to sde_fn
    """
    def __init__(self, dt, sde_fn, sde_pars, *args, **kwds):
        super().__init__(*args, **kwds)
        self.dt = dt = tt.as_tensor_variable(dt)
        self.sde_fn = sde_fn
        self.sde_pars = sde_pars

    def logp(self, x):
        xt = x[:-1]
        f, g = self.sde_fn(x[:-1], *self.sde_pars)
        mu = xt + self.dt * f
        sd = tt.sqrt(self.dt) * g
        return tt.sum(Normal.dist(mu=mu, sigma=sd).logp(x[1:]))

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        dt = dist.dt
        name = r'\text{%s}' % name
        return r'${} \sim \text{EulerMaruyama}(\mathit{{dt}}={})$'.format(name,
                                                get_variable_name(dt))



class MvGaussianRandomWalk(distribution.Continuous):
    R"""
    Multivariate Random Walk with Normal innovations

    Parameters
    ----------
    mu : tensor
        innovation drift, defaults to 0.0
    cov : tensor
        pos def matrix, innovation covariance matrix
    tau : tensor
        pos def matrix, inverse covariance matrix
    chol : tensor
        Cholesky decomposition of covariance matrix
    init : distribution
        distribution for initial value (Defaults to Flat())

    Notes
    -----
    Only one of cov, tau or chol is required.

    """
    def __init__(self, mu=0., cov=None, tau=None, chol=None, lower=True, init=Flat.dist(),
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.init = init
        self.innovArgs = (mu, cov, tau, chol, lower)
        self.innov = multivariate.MvNormal.dist(*self.innovArgs)
        self.mean = tt.as_tensor_variable(0.)

    def logp(self, x):
        x_im1 = x[:-1]
        x_i = x[1:]

        return self.init.logp_sum(x[0]) + self.innov.logp_sum(x_i - x_im1)

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        mu = dist.innov.mu
        cov = dist.innov.cov
        name = r'\text{%s}' % name
        return r'${} \sim \text{MvGaussianRandomWalk}(\mathit{{mu}}={},~\mathit{{cov}}={})$'.format(name,
                                                get_variable_name(mu),
                                                get_variable_name(cov))


class MvStudentTRandomWalk(MvGaussianRandomWalk):
    R"""
    Multivariate Random Walk with StudentT innovations

    Parameters
    ----------
    nu : degrees of freedom
    mu : tensor
        innovation drift, defaults to 0.0
    cov : tensor
        pos def matrix, innovation covariance matrix
    tau : tensor
        pos def matrix, inverse covariance matrix
    chol : tensor
        Cholesky decomposition of covariance matrix
    init : distribution
        distribution for initial value (Defaults to Flat())
    """
    def __init__(self, nu, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nu = tt.as_tensor_variable(nu)
        self.innov = multivariate.MvStudentT.dist(self.nu, None, *self.innovArgs)

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        nu = dist.innov.nu
        mu = dist.innov.mu
        cov = dist.innov.cov
        name = r'\text{%s}' % name
        return r'${} \sim \text{MvStudentTRandomWalk}(\mathit{{nu}}={},~\mathit{{mu}}={},~\mathit{{cov}}={})$'.format(name,
                                                get_variable_name(nu),
                                                get_variable_name(mu),
                                                get_variable_name(cov))
