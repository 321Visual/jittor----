# import torch.nn.utils.spectral_norm as spectral_norm
import jittor.nn as nn
import jittor

"""
Spectral Normalization from https://arxiv.org/abs/1802.05957
"""
import jittor
from jittor.misc import normalize
from typing import Any, Optional, TypeVar
from jittor.nn import Module


class SpectralNorm:
    _version: int = 1
    name: str
    dim: int
    n_power_iterations: int
    eps: float

    def __init__(self, name: str = 'weight', n_power_iterations: int = 1, dim: int = 0, eps: float = 1e-12) -> None:
        self.name = name
        self.dim = dim
        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but '
                             'got n_power_iterations={}'.format(n_power_iterations))
        self.n_power_iterations = n_power_iterations
        self.eps = eps

    def reshape_weight_to_matrix(self, weight: jittor.Var) -> jittor.Var:
        weight_mat = weight
        if self.dim != 0:
            # permute dim to front
            weight_mat = weight_mat.permute(self.dim,
                                            *[d for d in range(weight_mat.dim()) if d != self.dim])
        height = weight_mat.size(0)
        return weight_mat.reshape(height, -1)

    def compute_weight(self, module: Module, do_power_iteration: bool) -> jittor.Var:
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        v = getattr(module, self.name + '_v')
        weight_mat = self.reshape_weight_to_matrix(weight)

        if do_power_iteration:
            with jittor.no_grad():
                for _ in range(self.n_power_iterations):
                    # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                    # are the first left and right singular vectors.
                    # This power iteration produces approximations of `u` and `v`.
                    v = normalize(jittor.nn.matmul(weight_mat.transpose(), u), dim=0, eps=self.eps)
                    u = normalize(jittor.nn.matmul(weight_mat, v), dim=0, eps=self.eps)
                if self.n_power_iterations > 0:
                    # See above on why we need to clone
                    u = u.clone()
                    v = v.clone()
            if jittor.mpi:
                u = u.mpi_all_reduce("mean")
                v = v.mpi_all_reduce("mean")

        sigma = jittor.matmul(u, jittor.matmul(weight_mat, v))
        weight = weight / sigma
        return weight

    def __call__(self, module: Module, inputs: Any) -> None:
        # self.compute_weight(module, do_power_iteration=module.is_training())
        setattr(module, self.name, self.compute_weight(module, do_power_iteration=True
                                                       ))

    def _solve_v_and_rescale(self, weight_mat, u, target_sigma):
        v = jittor.matmul(jittor.matmul(weight_mat.transpose().mm(weight_mat).pinverse(),
                                        jittor.matmul(weight_mat.t(), u.unsqueeze(1)))).squeeze(1)
        return v.mul_(target_sigma / jittor.matmul(u, jittor.matmul(weight_mat, v)))

    @staticmethod
    def apply(module: Module, name: str, n_power_iterations: int, dim: int, eps: float) -> 'SpectralNorm':
        fn = SpectralNorm(name, n_power_iterations, dim, eps)
        weight = module._parameters[name]
        if weight is None:
            raise ValueError(f'`SpectralNorm` cannot be applied as parameter `{name}` is None')

        with jittor.no_grad():
            weight_mat = fn.reshape_weight_to_matrix(weight)

            h, w = weight_mat.size()
            u = normalize(jittor.randn([h]), dim=0, eps=fn.eps)
            v = normalize(jittor.randn([w]), dim=0, eps=fn.eps)

        delattr(module, fn.name)
        # module.register_parameter(fn.name + "_orig", weight)
        setattr(module, fn.name + "_orig", weight)
        setattr(module, fn.name, weight)
        setattr(module, fn.name + "_u", u)
        setattr(module, fn.name + "_v", v)
        module.register_pre_forward_hook(fn)
        return fn


T_module = TypeVar('T_module', bound=Module)


def spectral_norm(module: T_module,
                  name: str = 'weight',
                  n_power_iterations: int = 1,
                  eps: float = 1e-12,
                  dim: Optional[int] = None) -> T_module:
    r"""Applies spectral normalization to a parameter in the given module.

    .. math::
        \mathbf{W}_{SN} = \dfrac{\mathbf{W}}{\sigma(\mathbf{W})},
        \sigma(\mathbf{W}) = \max_{\mathbf{h}: \mathbf{h} \ne 0} \dfrac{\|\mathbf{W} \mathbf{h}\|_2}{\|\mathbf{h}\|_2}

    Spectral normalization stabilizes the training of discriminators (critics)
    in Generative Adversarial Networks (GANs) by rescaling the weight tensor
    with spectral norm :math:`\sigma` of the weight matrix calculated using
    power iteration method. If the dimension of the weight tensor is greater
    than 2, it is reshaped to 2D in power iteration method to get spectral
    norm. This is implemented via a hook that calculates spectral norm and
    rescales weight before every :meth:`~Module.forward` call.

    See `Spectral Normalization for Generative Adversarial Networks`_ .

    .. _`Spectral Normalization for Generative Adversarial Networks`: https://arxiv.org/abs/1802.05957

    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
        n_power_iterations (int, optional): number of power iterations to
            calculate spectral norm
        eps (float, optional): epsilon for numerical stability in
            calculating norms
        dim (int, optional): dimension corresponding to number of outputs,
            the default is ``0``, except for modules that are instances of
            ConvTranspose{1,2,3}d, when it is ``1``

    Returns:
        The original module with the spectral norm hook

    .. note::
        This function has been reimplemented as
        :func:`torch.nn.utils.parametrizations.spectral_norm` using the new
        parametrization functionality in
        :func:`torch.nn.utils.parametrize.register_parametrization`. Please use
        the newer version. This function will be deprecated in a future version
        of PyTorch.

    Example::

        >>> m = spectral_norm(nn.Linear(20, 40))
        >>> m
        Linear(in_features=20, out_features=40, bias=True)
        >>> m.weight_u.size()
        torch.Size([40])

    """
    if dim is None:
        if isinstance(module, (jittor.nn.ConvTranspose,
                               jittor.nn.ConvTranspose3d)):
            dim = 1
        else:
            dim = 0
    SpectralNorm.apply(module, name, n_power_iterations, dim, eps)
    return module


def get_spectral_norm(opt):
    if opt.no_spectral_norm:
        return nn.Identity()
    else:
        return spectral_norm  # todo 使用spectral_norm 进行模型优化


# class SpectralNorm:
#     def __init__(self, name='weight', power_iterations=1):
#         self.name = name
#         self.power_iterations = power_iterations
#
#     @staticmethod
#     def l2normalize(v, eps=1e-12):
#         return v / (v.norm() + eps)
#
#     @staticmethod
#     def apply(module, name):
#         fn = SpectralNorm(name)
#         weight = getattr(module, name)
#         height = weight.shape[0]
#         width = weight.view(height, -1).shape[1]
#         u = jittor.normal(mean=0, std=1, dtype=weight.dtype, size=height)
#         u.requires_grad = False
#         v = jittor.normal(mean=0, std=1, size=width, dtype=weight.dtype)
#         v.requires_grad = False
#         u.data = SpectralNorm.l2normalize(u)
#         v.data = SpectralNorm.l2normalize(v)
#         delattr(module, name)
#         setattr(module, "_" + name + "_u", u)
#         setattr(module, "_" + name + "_v", v)
#         setattr(module, name, weight)
#         module.register_pre_forward_hook(fn)
#         return fn
#
#     def compute_weight(self, module):
#         u = getattr(module, "_" + self.name + "_u")
#         v = getattr(module, "_" + self.name + "_v")
#         w = getattr(module, self.name)
#
#         height = w.shape[0]
#         for _ in range(self.power_iterations):
#             v = self.l2normalize(jittor.matmul(jittor.transpose(w.view(height, -1)), u))
#             u = self.l2normalize(jittor.matmul(w.view(height, -1), v))
#         sigma = u.matmul(w.view(height, -1).matmul(v))
#         return w / sigma.expand_as(w)
#
#     def __call__(self, module, input):
#         weight = self.compute_weight(module)
#         setattr(module, self.name, weight)


class SPADE(nn.Module):
    def __init__(self, opt, norm_nc, label_nc):
        super().__init__()
        self.first_norm = get_norm_layer(opt, norm_nc)
        ks = opt.spade_ks
        nhidden = 128
        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def execute(self, x, segmap):
        normalized = self.first_norm(x)
        segmap = nn.interpolate(segmap, size=x.shape[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = normalized * (1 + gamma) + beta
        return out


def get_norm_layer(opt, norm_nc):
    if opt.param_free_norm == 'instance':
        return nn.InstanceNorm2d(norm_nc, affine=False)
    if opt.param_free_norm == 'batch':
        return nn.BatchNorm2d(norm_nc, affine=False)
    else:
        raise ValueError('%s is not a recognized param-free norm type in SPADE'
                         % opt.param_free_norm)
