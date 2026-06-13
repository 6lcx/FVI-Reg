import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F

from Modules.Loss import LOSSDICT
from .REG_ConvBlock import MixVisionTransformer_FF, SegFormerHead
from .BaseNetwork import GenerativeRegistrationNetwork
from .SegFormer_GSFVI_ConvBlock import MixVisionTransformer_GSFVI, BayesConv2d, BayesLinear
from .SegFormerRBF_ConvBlock import MixVisionTransformer
from torch.nn.modules.utils import _pair



class BayesianConv2d(nn.Module):
    def __init__(
        self,in_channels,out_channels,kernel_size,stride=1,padding=0,dilation=1,groups=1,bias=True,
        prior_mu=0.0,prior_sigma=1.0,posterior_mu_init=0.0,posterior_rho_init=-5.0,
        rank=4,u_init_std=1e-4,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.use_bias = bias

        self.prior_mu = float(prior_mu)
        self.prior_sigma = float(prior_sigma)
        self.posterior_mu_init = float(posterior_mu_init)
        self.posterior_rho_init = float(posterior_rho_init)
        self.lowrank_rank = int(rank)
        self.lowrank_u_init_std = float(u_init_std)
        self._logvar_eps = 1e-12
        weight_shape = (out_channels, in_channels // groups, *self.kernel_size)
        weight_param_dim = (in_channels // groups) * self.kernel_size[0] * self.kernel_size[1]
        self.parameter_dim = weight_param_dim + (1 if self.use_bias else 0)
        self.weight_mu = nn.Parameter(torch.empty(weight_shape))
        self.weight_rho = nn.Parameter(torch.empty(weight_shape))

        if self.use_bias:
            self.bias_mu = nn.Parameter(torch.empty(out_channels))
            self.bias_rho = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias_mu", None)
            self.register_parameter("bias_rho", None)

        self.lowrank_U = nn.Parameter(torch.empty(out_channels, self.parameter_dim, self.lowrank_rank))

        self.reset_parameters()

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        lowrank_key = prefix + "lowrank_U"
        if lowrank_key not in state_dict:
            state_dict[lowrank_key] = self.lowrank_U.detach().clone()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def reset_parameters(self):
        if self.posterior_mu_init == 0.0:
            nn.init.kaiming_uniform_(self.weight_mu, a=5 ** 0.5)
        else:
            nn.init.normal_(self.weight_mu, mean=self.posterior_mu_init, std=0.1)

        nn.init.constant_(self.weight_rho, self.posterior_rho_init)

        if self.use_bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
            bound = (1.0 / fan_in) ** 0.5 if fan_in > 0 else 0.0
            if self.posterior_mu_init == 0.0:
                nn.init.uniform_(self.bias_mu, -bound, bound)
            else:
                nn.init.normal_(self.bias_mu, mean=self.posterior_mu_init, std=0.1)

            nn.init.constant_(self.bias_rho, self.posterior_rho_init)

        nn.init.normal_(self.lowrank_U, mean=0.0, std=self.lowrank_u_init_std)

    @staticmethod
    def _rho_to_sigma(rho):
        return F.softplus(rho)

    @staticmethod
    def _resolve_sample_flag(sample=None, mode=None, default=True):
        if mode is not None:
            sample = mode

        if sample is None:
            return bool(default)
        if isinstance(sample, bool):
            return sample
        if sample == "sample":
            return True
        return sample != "mean"

    def posterior_std(self):
        stats = {"weight": self._rho_to_sigma(self.weight_rho)}
        if self.use_bias:
            stats["bias"] = self._rho_to_sigma(self.bias_rho)
        return stats

    def posterior_var(self):
        std = self.posterior_std()
        stats = {"weight": std["weight"].square()}
        if "bias" in std:
            stats["bias"] = std["bias"].square()
        return stats

    def posterior_logvars(self):
        var = self.posterior_var()
        stats = {"weight": torch.log(var["weight"].clamp_min(self._logvar_eps))}
        if "bias" in var:
            stats["bias"] = torch.log(var["bias"].clamp_min(self._logvar_eps))
        return stats

    @property
    def weight_logvar(self):
        return self.posterior_logvars()["weight"]

    @property
    def bias_logvar(self):
        if not self.use_bias:
            return None
        return self.posterior_logvars()["bias"]

    def _flatten_parameter_parts(self, weight, bias=None):
        flat = weight.reshape(self.out_channels, -1)
        if bias is not None:
            flat = torch.cat((flat, bias.reshape(self.out_channels, 1)), dim=1)
        return flat

    def _split_parameter_vector(self, parameter_vector):
        weight_dim = self.weight_mu[0].numel()
        weight = parameter_vector[:, :weight_dim].reshape_as(self.weight_mu)
        bias = parameter_vector[:, weight_dim] if self.use_bias else None
        return weight, bias

    def sample_parameters(self, return_stats=False):
        weight_sigma = self._rho_to_sigma(self.weight_rho)
        bias_sigma = self._rho_to_sigma(self.bias_rho) if self.use_bias else None

        parameter_mu = self._flatten_parameter_parts(self.weight_mu, self.bias_mu if self.use_bias else None)
        parameter_sigma = self._flatten_parameter_parts(weight_sigma, bias_sigma)
        eps_diag = torch.randn_like(parameter_mu)
        eps_rank = torch.randn(
            self.out_channels,
            self.lowrank_rank,
            device=parameter_mu.device,
            dtype=parameter_mu.dtype,
        )
        lowrank_sample = torch.matmul(self.lowrank_U, eps_rank.unsqueeze(-1)).squeeze(-1)
        parameter_sample = parameter_mu + parameter_sigma * eps_diag + lowrank_sample
        weight, bias = self._split_parameter_vector(parameter_sample)

        if not return_stats:
            return weight, bias

        return {
            "weight": weight,
            "bias": bias,
            "weight_sigma": weight_sigma,
            "bias_sigma": bias_sigma,
        }

    def forward_mean(self, x):
        bias = self.bias_mu if self.use_bias else None
        return F.conv2d(
            x,
            self.weight_mu,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def forward_sample(self, x):
        weight, bias = self.sample_parameters(return_stats=False)
        return F.conv2d(
            x,
            weight,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def _gaussian_kl(self, mu_q, logvar_q):
        prior_mu = mu_q.new_tensor(self.prior_mu)
        prior_var = mu_q.new_tensor(self.prior_sigma ** 2)
        var_q = torch.exp(logvar_q)
        kl = 0.5 * (
            torch.log(prior_var) - logvar_q + (var_q + (mu_q - prior_mu).pow(2)) / prior_var - 1.0
        )
        return kl.sum()

    def kl_divergence(self):
        weight_kl = self._gaussian_kl(self.weight_mu, self.weight_logvar)
        bias_kl = self.weight_mu.new_zeros(())
        if self.use_bias:
            bias_kl = self._gaussian_kl(self.bias_mu, self.bias_logvar)
        return weight_kl + bias_kl

    def kl_loss(self):
        return self.kl_divergence()

    def forward(self, x, sample=None, return_kl=False, mode=None):
        sample = self._resolve_sample_flag(sample=sample, mode=mode, default=self.training)
        output = self.forward_sample(x) if sample else self.forward_mean(x)

        if return_kl:
            return output, self.kl_divergence()
        return output


class FVIBlockDiagonalPosterior:

    @staticmethod
    def _build_single_channel_cov(design, parameter_var, lowrank_u, jitter=1e-6):
        """Build one full spatial covariance block.

        Args:
            design: [B, P, D_w] unfolded feature patches, with a trailing
                ones column when the final BayesianConv2d has bias.
            parameter_var: [D_w] diagonal posterior variance for one output channel.
            lowrank_u: [D_w, rank] low-rank posterior covariance factor.
        """
        batch_size, num_positions, parameter_dim = design.shape

        design_weighted = design * parameter_var.view(1, 1, -1)
        cov = torch.bmm(design_weighted, design.transpose(1, 2))

        # Z: [B, P, rank], K_lowrank: [B, P, P].
        z = torch.matmul(design, lowrank_u)
        k_lowrank = torch.bmm(z, z.transpose(1, 2))
        cov = cov + k_lowrank

        cov = 0.5 * (cov + cov.transpose(1, 2))

        if jitter is not None and jitter > 0:
            eye = torch.eye(cov.size(1), device=cov.device, dtype=cov.dtype).unsqueeze(0)
            cov = cov + jitter * eye

        return 0.5 * (cov + cov.transpose(1, 2))

    @staticmethod
    def _cov_to_spectral_form(cov):
        """Return U and lambda for cov = U diag(lambda) U^T."""
        cov = 0.5 * (cov + cov.transpose(1, 2))
        input_dtype = cov.dtype
        # CUDA float32 eigh can fail on the posterior blocks because these
        # PSD-plus-jitter matrices may be ill-conditioned and have clustered
        # eigenvalues. Decompose in float64, then cast back for downstream use.
        lambdas, u = torch.linalg.eigh(cov.to(torch.float64))
        lambdas = torch.flip(lambdas, dims=(1,)).clamp_min(0.0)
        u = torch.flip(u, dims=(2,))
        return u.to(input_dtype), lambdas.to(input_dtype)

    @staticmethod
    def _cov_to_lowrank_spectral_form(cov, rank, oversample=8, niter=2):
        """Approximate cov = U diag(lambda) U^T with only the leading rank terms."""
        cov = 0.5 * (cov + cov.transpose(1, 2))
        input_dtype = cov.dtype
        rank = max(1, min(int(rank), cov.size(1)))
        q = max(rank, min(cov.size(1), rank + int(oversample)))

        u_list = []
        lambda_list = []
        for b in range(cov.size(0)):
            u_b, s_b, _ = torch.svd_lowrank(
                cov[b].to(torch.float64),
                q=q,
                niter=niter,
            )
            order = torch.argsort(s_b, descending=True)
            order = order[:rank]
            u_list.append(u_b[:, order].to(input_dtype))
            lambda_list.append(s_b[order].clamp_min(0.0).to(input_dtype))

        return torch.stack(u_list, dim=0), torch.stack(lambda_list, dim=0)

    @staticmethod
    def _assemble_block_diagonal(cov_xx, cov_yy):
        zero = torch.zeros_like(cov_xx)
        upper = torch.cat((cov_xx, zero), dim=2)
        lower = torch.cat((zero, cov_yy), dim=2)
        cov = torch.cat((upper, lower), dim=1)
        return 0.5 * (cov + cov.transpose(1, 2))

    @staticmethod
    def _assemble_block_diagonal_spectral(u_x, lambda_x, u_y, lambda_y):
        batch_size, num_positions, rank_x = u_x.shape
        _, num_positions_y, rank_y = u_y.shape
        zero_x = torch.zeros(
            batch_size,
            num_positions,
            rank_y,
            device=u_x.device,
            dtype=u_x.dtype,
        )
        zero_y = torch.zeros(
            batch_size,
            num_positions,
            rank_x,
            device=u_y.device,
            dtype=u_y.dtype,
        )
        upper = torch.cat((u_x, zero_x), dim=2)
        lower = torch.cat((zero_y, u_y), dim=2)
        full_u = torch.cat((upper, lower), dim=1)
        full_lambda = torch.cat((lambda_x, lambda_y), dim=1)
        return full_u, full_lambda

    @staticmethod
    def build(last_layer, feature_map, jitter=1e-6, kl_jitter=0.0, spectral_rank=None):
        """Construct q(alpha | feature_map) from the last Bayesian Conv2d.

        Returns:
            alpha_mean_map: [B, 2, H, W]
            alpha_cov: [B, 2HW, 2HW]
        """
        alpha_mean_map = last_layer.forward_mean(feature_map)
        batch_size = feature_map.size(0)
        _, _, out_h, out_w = alpha_mean_map.shape
        num_positions = out_h * out_w

        # unfolded: [B, Cin*kH*kW, HW]
        # design:   [B, HW, Cin*kH*kW]
        unfolded = F.unfold(
            feature_map,
            kernel_size=last_layer.kernel_size,
            dilation=last_layer.dilation,
            padding=last_layer.padding,
            stride=last_layer.stride,
        )
        design = unfolded.transpose(1, 2).contiguous()
        if last_layer.use_bias:
            ones = design.new_ones(design.size(0), design.size(1), 1)
            design = torch.cat((design, ones), dim=2)

        posterior_var = last_layer.posterior_var()
        weight_var = posterior_var["weight"].view(last_layer.out_channels, -1)
        if last_layer.use_bias:
            bias_var = posterior_var["bias"].view(last_layer.out_channels, 1)
            parameter_var = torch.cat((weight_var, bias_var), dim=1)
        else:
            parameter_var = weight_var

        cov_xx = FVIBlockDiagonalPosterior._build_single_channel_cov(
            design=design,
            parameter_var=parameter_var[0],
            lowrank_u=last_layer.lowrank_U[0],
            jitter=jitter,
        )
        cov_yy = FVIBlockDiagonalPosterior._build_single_channel_cov(
            design=design,
            parameter_var=parameter_var[1],
            lowrank_u=last_layer.lowrank_U[1],
            jitter=jitter,
        )
        if kl_jitter is not None and kl_jitter > 0:
            eye = torch.eye(cov_xx.size(1), device=cov_xx.device, dtype=cov_xx.dtype).unsqueeze(0)
            spectral_cov_x = cov_xx + kl_jitter * eye
            spectral_cov_y = cov_yy + kl_jitter * eye
        else:
            spectral_cov_x = cov_xx
            spectral_cov_y = cov_yy
        if spectral_rank is None:
            spectral_u_x, spectral_lambda_x = FVIBlockDiagonalPosterior._cov_to_spectral_form(spectral_cov_x)
            spectral_u_y, spectral_lambda_y = FVIBlockDiagonalPosterior._cov_to_spectral_form(spectral_cov_y)
        else:
            spectral_u_x, spectral_lambda_x = FVIBlockDiagonalPosterior._cov_to_lowrank_spectral_form(
                spectral_cov_x,
                rank=spectral_rank,
            )
            spectral_u_y, spectral_lambda_y = FVIBlockDiagonalPosterior._cov_to_lowrank_spectral_form(
                spectral_cov_y,
                rank=spectral_rank,
            )
        spectral_u_full, spectral_lambda_full = FVIBlockDiagonalPosterior._assemble_block_diagonal_spectral(
            spectral_u_x,
            spectral_lambda_x,
            spectral_u_y,
            spectral_lambda_y,
        )
        alpha_cov = FVIBlockDiagonalPosterior._assemble_block_diagonal(cov_xx, cov_yy)
        result = {
            "alpha_mean_map": alpha_mean_map,
            "alpha_cov": alpha_cov,
            "alpha_cov_spectral": {
                "x": {
                    "u": spectral_u_x,
                    "lambda": spectral_lambda_x,
                },
                "y": {
                    "u": spectral_u_y,
                    "lambda": spectral_lambda_y,
                },
                "full": {
                    "u": spectral_u_full,
                    "lambda": spectral_lambda_full,
                },
                "represents": "posterior covariance used by KL after qalpha jitter and explicit KL jitter",
                "qalpha_jitter": jitter,
                "kl_jitter": kl_jitter,
            },
            "feature_map": feature_map,
            "meta": {
                "feature_map_shape": tuple(feature_map.shape),
                "alpha_mean_map_shape": tuple(alpha_mean_map.shape),
                "num_positions": num_positions,
                "output_dim": 2 * num_positions,
                "design_shape": tuple(design.shape),
                "lowrank_u_shape": tuple(last_layer.lowrank_U.shape),
                "lowrank_rank": last_layer.lowrank_rank,
                "spectral_rank_x": int(spectral_lambda_x.shape[1]),
                "spectral_rank_y": int(spectral_lambda_y.shape[1]),
                "spectral_rank_full": int(spectral_lambda_full.shape[1]),
                "covariance_shape": tuple(alpha_cov.shape),
                "jitter": jitter,
                "kl_jitter": kl_jitter,
            },
        }
        return result



class SegFormer(nn.Module):
    def __init__(self, num_classes=21,
                 mit_embed_dims=[32, 64, 160, 256],
                 mit_depths=[2, 2, 2, 2],
                 mit_num_heads=[1, 2, 5, 8],
                 mit_sr_ratios=[8, 4, 2, 1],
                 in_chans=1,
                 drop_rate=0.0,
                 drop_path_rate=0.0):
        super(SegFormer, self).__init__()
        self.in_channels = [32, 64, 160, 256]
        self.backbone = MixVisionTransformer_FF(
            patch_size=4, in_chans=in_chans,
            embed_dims=mit_embed_dims, num_heads=mit_num_heads,
            mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=mit_depths, sr_ratios=mit_sr_ratios,
            drop_rate=drop_rate, drop_path_rate=drop_path_rate
        )
        self.embedding_dim = 256
        self.decode_head = SegFormerHead(num_classes, self.in_channels, self.embedding_dim)

    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)

        x = self.backbone.forward(inputs)
        x = self.decode_head.forward(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x

class SegFormerRBFInterEncoder(nn.Module):
    def __init__(self,
                 mit_embed_dims=[32, 64, 160, 256],
                 mit_depths=[2, 2, 2, 2],
                 mit_num_heads=[1, 2, 5, 8],
                 mit_sr_ratios=[8, 4, 2, 1],
                 in_chans=2,
                 drop_rate=0.0,
                 drop_path_rate=0.0,
                 **_):
        super().__init__()
        self.backbone = MixVisionTransformer(
            in_chans=in_chans,
            embed_dims=mit_embed_dims,
            num_heads=mit_num_heads,
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=mit_depths,
            sr_ratios=mit_sr_ratios,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate
        )

        _, _, C3, _ = mit_embed_dims

        self.head = nn.Sequential(
            nn.Conv2d(C3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.alpha = nn.Conv2d(16, 2, 3, 1, 1)


    def extract_alpha_features(self, src, tgt):
        x_in = torch.cat([src, tgt], dim=1)
        _, _, F3, _ = self.backbone(x_in)
        return self.head(F3)

    def forward_alpha(self, alpha_features, alpha_mode="mean"):
        if isinstance(self.alpha, BayesianConv2d):
            return self.alpha(alpha_features, sample=alpha_mode)
        return self.alpha(alpha_features)

    def forward(self, src, tgt, alpha_mode="mean"):
        dre = self.extract_alpha_features(src, tgt)
        return self.forward_alpha(dre, alpha_mode=alpha_mode)

class SegFormer_GSFVI_InterEncoder(nn.Module):
    def __init__(self,
                 mit_embed_dims=[32, 64, 160, 256],
                 mit_depths=[2, 2, 2, 2],
                 mit_num_heads=[1, 2, 5, 8],
                 mit_sr_ratios=[8, 4, 2, 1],
                 in_chans=2,
                 drop_rate=0.0,
                 drop_path_rate=0.0,
                 **_):
        super().__init__()
        self.backbone = MixVisionTransformer_GSFVI(
            in_chans=in_chans,
            embed_dims=mit_embed_dims,
            num_heads=mit_num_heads,
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=mit_depths,
            sr_ratios=mit_sr_ratios,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate
        )

        _, _, C3, _ = mit_embed_dims

        self.head = nn.Sequential(
            BayesConv2d(C3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            BayesConv2d(64, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.qalpha = BayesianConv2d(16, 2, 3, 1, 1)

    @staticmethod
    def _resolve_sample_flag(sample=None, default=True):
        return BayesianConv2d._resolve_sample_flag(sample=sample, default=default)

    def _forward_head(self, feature_map, sample=None):
        sample = self._resolve_sample_flag(sample=sample, default=self.training)

        x = feature_map
        for module in self.head:
            if isinstance(module, (BayesianConv2d, BayesLinear, BayesConv2d)):
                x = module(x, sample=sample)
            else:
                x = module(x)
        return x

    def extract_qalpha_features(self, src, tgt, sample=True):
        x_in = torch.cat([src, tgt], dim=1)
        _, _, feature_stage3, _ = self.backbone(x_in)
        return self._forward_head(feature_stage3, sample=sample)

    def forward_qalpha(self, qalpha_features, qalpha_mode=None):
        return self.qalpha(qalpha_features, sample=qalpha_mode)

    def forward_posterior_moments_from_features(
        self,
        qalpha_features,
        cov_jitter=1e-6,
        kl_jitter=0.0,
        spectral_rank=None,
    ):
        posterior = FVIBlockDiagonalPosterior.build(
            last_layer=self.qalpha,
            feature_map=qalpha_features,
            jitter=cov_jitter,
            kl_jitter=kl_jitter,
            spectral_rank=spectral_rank,
        )
        posterior["qalpha_features"] = qalpha_features
        return posterior

    def forward(self, src, tgt, qalpha_mode=None, sample_upstream=None, return_features=False):
        qalpha_features = self.extract_qalpha_features(src,tgt,
                        sample=self._resolve_sample_flag(sample=sample_upstream, default=self.training))
        alpha_map = self.forward_qalpha(qalpha_features, qalpha_mode=qalpha_mode)
        if return_features:
            return alpha_map, qalpha_features
        return alpha_map


class RadialBasisLayer(nn.Module):
    def __init__(self, cpoint_pos, img_size=(128, 128), c=2):
        super(RadialBasisLayer, self).__init__()

        cpoint_num = cpoint_pos.size()[0]

        # Build the dense image lattice in the same coordinate system as the control lattice.
        loc_vectors = [torch.linspace(0.0, 15.0, i_s) for i_s in img_size]
        loc = torch.meshgrid(loc_vectors)
        loc = torch.stack(loc, 2)
        loc = loc[:, :, [1, 0]].float().unsqueeze(2)

        loc_tile = loc.repeat(1, 1, cpoint_num, 1)

        cp_loc = cpoint_pos.unsqueeze(0).unsqueeze(0)
        cp_loc_tile = cp_loc.repeat(*img_size, 1, 1)

        dist = torch.norm(loc_tile - cp_loc_tile, dim=3) / c

        mask = dist < 1

        # Precompute compact-support Wendland weights for every pixel/control-point pair.
        weight = torch.pow(1 - dist, 4) * (4 * dist + 1)    # (H,W,N)
        weight = weight * mask.float()                      # (H,W,N)

        weight = weight.unsqueeze(0).unsqueeze(4)

        self.register_buffer('weight', weight)

    def forward(self, alpha):
        alpha = alpha.unsqueeze(1).unsqueeze(1)

        # Combine sparse control-point displacements into a dense deformation field.
        # (1,H,W,N,1) * (B,1,1,N,2) -> (B,H,W,N,2) -> sum_N -> (B,H,W,2)
        phi = torch.sum(self.weight * alpha, 3)

        # Match the transformer layout.
        phi = phi.permute(0, 3, 1, 2)
        return phi


class RBFBendingEnergyLoss(nn.Module):
    def __init__(self, cpoint_pos, r):
        super(RBFBendingEnergyLoss, self).__init__()
        self.num_cp = cpoint_pos.size()[0]
        scppos = cpoint_pos.unsqueeze(1).repeat(1, self.num_cp, 1)
        despos = cpoint_pos.unsqueeze(0).repeat(self.num_cp, 1, 1)
        dis = torch.norm(scppos - despos, dim=2) / r
        filter_dis = dis < 1
        # Reuse the same compact-support kernel to penalize non-smooth control-point motion.
        weight = torch.pow(1 - dis, 4) * (4 * dis + 1)
        weight = (weight * filter_dis.float()).unsqueeze(0)
        self.register_buffer('weight', weight)

    def be(self, alpha):
        flatted_alpha = torch.flatten(alpha, start_dim=1)
        tiled_alpha = flatted_alpha.unsqueeze(1).repeat(1, self.num_cp, 1)
        temp_res = torch.sum(tiled_alpha * self.weight, dim=2)
        be = torch.sum(flatted_alpha * temp_res, dim=1)
        return be

    def forward(self, alpha):
        be_x = self.be(alpha[:, :, 0])
        be_y = self.be(alpha[:, :, 1])
        return (be_x + be_y) / 2


class SegFormer_GSFVIGenerativeNetwork(GenerativeRegistrationNetwork):
    def __init__(
        self,
        encoder_param,
        c=2,
        i_size=None,
        similarity_factor=150000,
        similarity_loss='LCC',
        similarity_loss_param=None,
        prior_l_feat=0.75,
        prior_l_space=0.5,
        prior_sigma_f=1.0,
        prior_eps=1e-4,
        posterior_kl_weight=1e-5,
        posterior_kl_jitter=1e-5,
        prior_kl_lowrank_rank=10,
        posterior_kl_lowrank_rank=10,
        qalpha_train_mode="sample",
        qalpha_eval_mode="mean",
        qalpha_linearization_jitter=1e-5,
        train_mc_samples=4,
        test_mc_samples=16,
        **_,
    ):
        if i_size is None:
            i_size = [128, 128]
        if similarity_loss_param is None:
            similarity_loss_param = {}
        super(SegFormer_GSFVIGenerativeNetwork, self).__init__(i_size)

        cp_loc_vectors = [torch.arange(0, s) for s in (16, 16)]
        cp_loc = torch.meshgrid(cp_loc_vectors)
        cp_loc = torch.stack(cp_loc, 2)[:, :, [1, 0]]
        cp_loc = torch.flatten(cp_loc, start_dim=0, end_dim=1).float()
        prior_cp_loc_norm = 2.0 * cp_loc / 15.0 - 1.0
        self.register_buffer('prior_cp_loc_norm', prior_cp_loc_norm)

        self.encoder = SegFormer_GSFVI_InterEncoder(**encoder_param)

        self.decoder = RadialBasisLayer(cp_loc, i_size, c)

        self.similarity_factor = similarity_factor
        self.similarity_loss = LOSSDICT[similarity_loss](**similarity_loss_param)
        self.bending_energy = RBFBendingEnergyLoss(cp_loc, c)

        self.seg_F = SegFormer(2, mit_embed_dims=[32, 64, 160, 256],
                               mit_depths=[2, 2, 2, 2],
                               mit_num_heads=[1, 2, 5, 8],
                               mit_sr_ratios=[8, 4, 2, 1],
                               in_chans=1,
                               drop_rate=0.0,
                               drop_path_rate=0.0)
        self.reg_pretrain = SegFormerRBFInterEncoder(**encoder_param)
        self.prior_l_feat = prior_l_feat
        self.prior_l_space = prior_l_space
        self.prior_sigma_f = prior_sigma_f
        self.prior_eps = prior_eps
        self.posterior_kl_weight = posterior_kl_weight
        self.posterior_kl_jitter = posterior_kl_jitter
        self.prior_kl_lowrank_rank = int(prior_kl_lowrank_rank)
        self.posterior_kl_lowrank_rank = int(posterior_kl_lowrank_rank)
        self.qalpha_train_mode = qalpha_train_mode
        self.qalpha_eval_mode = qalpha_eval_mode
        self.qalpha_linearization_jitter = qalpha_linearization_jitter
        self.train_mc_samples = int(train_mc_samples)
        self.test_mc_samples = int(test_mc_samples)

        for p in self.seg_F.parameters():
            p.requires_grad_(False)
        for p in self.reg_pretrain.parameters():
            p.requires_grad_(False)

        name = str(similarity_loss) + '--'
        for k in similarity_loss_param:
            name += '-' + str(similarity_loss_param[k])
        name += '--' + str(similarity_factor)
        self.name = name

    @staticmethod
    def _flatten_alpha(alpha_map):
        # Point-major control-point order for the decoder: [(x_1,y_1), ..., (x_N,y_N)].
        return torch.flatten(alpha_map, start_dim=2).permute(0, 2, 1)

    @staticmethod
    def _flatten_alpha_xy(alpha_map):
        # Channel-major flatten for Gaussian vectors/covariances: [x_1,...,x_N,y_1,...,y_N].
        alpha_x = torch.flatten(alpha_map[:, 0], start_dim=1)
        alpha_y = torch.flatten(alpha_map[:, 1], start_dim=1)
        return torch.cat((alpha_x, alpha_y), dim=1)

    def _resolve_qalpha_mode(self, qalpha_mode=None):
        if qalpha_mode is not None:
            return qalpha_mode
        return self.qalpha_train_mode if self.training else self.qalpha_eval_mode

    def _resolve_upstream_sampling(self, sample_upstream=None):
        return BayesianConv2d._resolve_sample_flag(sample=sample_upstream, default=self.training)

    @staticmethod
    def _pairwise_square_dist(x):
        x_norm = torch.sum(x * x, dim=2, keepdim=True)
        dist = x_norm + x_norm.transpose(1, 2) - 2.0 * torch.matmul(x, x.transpose(1, 2))
        return dist.clamp_min(0.0)

    def _prior_cp_loc_for_batch(self, batch_size):
        return self.prior_cp_loc_norm.unsqueeze(0).repeat(batch_size, 1, 1)

    @staticmethod
    def _feature_map_to_prior_cp_feature(feat):
        if feat.shape[-2:] != (16, 16):
            feat = F.interpolate(feat, size=(16, 16), mode="bilinear", align_corners=False)
        cp_feat = torch.flatten(feat, start_dim=2).permute(0, 2, 1)
        return F.normalize(cp_feat, p=2, dim=2, eps=1e-12)

    def _build_prior_space_cov(self, cp_feat, cp_loc):
        loc_dist = self._pairwise_square_dist(cp_loc)

        k_loc = torch.exp(-loc_dist / (2.0 * self.prior_l_space * self.prior_l_space))
        feat_dist = self._pairwise_square_dist(cp_feat)
        k_feat = torch.exp(-feat_dist / (2.0 * self.prior_l_feat * self.prior_l_feat))

        k_space = (self.prior_sigma_f ** 2) * k_feat * k_loc
        k_space = 0.5 * (k_space + k_space.transpose(1, 2))

        eye = torch.eye(k_space.size(1), device=k_space.device, dtype=k_space.dtype).unsqueeze(0)
        k_space = k_space + self.prior_eps * eye
        k_space = 0.5 * (k_space + k_space.transpose(1, 2))
        return k_space

    @staticmethod
    def _expand_prior_cov_xy(k_space):
        zero = torch.zeros_like(k_space)
        upper = torch.cat((k_space, zero), dim=2)
        lower = torch.cat((zero, k_space), dim=2)
        return torch.cat((upper, lower), dim=1)

    def build_prior_mean(self, src, tgt):
        reg_flag = self.reg_pretrain.training
        self.reg_pretrain.eval()
        with torch.no_grad():
            prior_alpha = self.reg_pretrain(src, tgt, alpha_mode="mean")
            prior_mean = self._flatten_alpha_xy(prior_alpha)
        self.reg_pretrain.train(reg_flag)
        return prior_mean

    def build_prior_cov(self, tgt, return_aux=False):
        seg_flag = self.seg_F.training
        self.seg_F.eval()
        with torch.no_grad():
            feature_stage3 = self.seg_F.backbone.forward(tgt)[2]
            cp_feat = self._feature_map_to_prior_cp_feature(feature_stage3)
            cp_loc = self._prior_cp_loc_for_batch(tgt.size(0))
            k_space = self._build_prior_space_cov(cp_feat.float(), cp_loc.float())
            prior_cov = self._expand_prior_cov_xy(k_space)
        self.seg_F.train(seg_flag)

        if return_aux:
            return prior_cov, cp_feat, cp_loc, k_space
        return prior_cov

    def build_prior(self, src, tgt, return_aux=False):
        prior_mean = self.build_prior_mean(src, tgt)
        prior_cov, cp_feat, cp_loc, k_space = self.build_prior_cov(
            tgt, return_aux=True
        )

        if return_aux:
            return prior_mean, prior_cov, cp_feat, cp_loc, k_space
        return prior_mean, prior_cov

    @staticmethod
    def _batched_lowrank_residual_diag_decomposition(cov, rank, eps=1e-8):
        cov = 0.5 * (cov + cov.transpose(1, 2))
        rank = max(1, min(int(rank), cov.size(1)))
        top_vecs, top_vals = FVIBlockDiagonalPosterior._cov_to_lowrank_spectral_form(
            cov,
            rank=rank,
        )
        factor = top_vecs * torch.sqrt(top_vals.clamp_min(0.0)).unsqueeze(1)
        lowrank = factor @ factor.transpose(1, 2)
        residual_diag = torch.diagonal(cov - lowrank, dim1=1, dim2=2).clamp_min(eps)
        return residual_diag, factor

    @staticmethod
    def _assemble_blockdiag_factor(factor_x, factor_y):
        batch_size, block_dim, rank_x = factor_x.shape
        _, block_dim_y, rank_y = factor_y.shape
        zero_x = torch.zeros(batch_size, block_dim, rank_y, device=factor_x.device, dtype=factor_x.dtype)
        zero_y = torch.zeros(batch_size, block_dim, rank_x, device=factor_y.device, dtype=factor_y.dtype)
        upper = torch.cat((factor_x, zero_x), dim=2)
        lower = torch.cat((zero_y, factor_y), dim=2)
        return torch.cat((upper, lower), dim=1)

    @staticmethod
    def _woodbury_solve(diag, factor, rhs):
        diag_inv = diag.reciprocal()
        d_inv_rhs = diag_inv.unsqueeze(-1) * rhs
        d_inv_factor = diag_inv.unsqueeze(-1) * factor
        eye = torch.eye(factor.size(2), device=factor.device, dtype=factor.dtype).unsqueeze(0)
        middle = eye + factor.transpose(1, 2) @ d_inv_factor
        rhs_small = factor.transpose(1, 2) @ d_inv_rhs
        correction = d_inv_factor @ torch.linalg.solve(middle, rhs_small)
        return d_inv_rhs - correction

    @staticmethod
    def _batched_paper_lowrank_kl(
        q_mean,
        posterior_spectral,
        p_mean,
        p_cov,
        prior_rank,
        posterior_rank,
        kl_jitter,
        eps=1e-8,
        return_components=False,
    ):
        batch_size, dim = q_mean.shape
        block_dim = dim // 2

        eye_block = torch.eye(block_dim, device=p_cov.device, dtype=p_cov.dtype).unsqueeze(0)
        p_cov_x = 0.5 * (p_cov[:, :block_dim, :block_dim] + p_cov[:, :block_dim, :block_dim].transpose(1, 2))
        p_cov_y = 0.5 * (p_cov[:, block_dim:, block_dim:] + p_cov[:, block_dim:, block_dim:].transpose(1, 2))
        if kl_jitter is not None and kl_jitter > 0:
            p_cov_x = p_cov_x + kl_jitter * eye_block
            p_cov_y = p_cov_y + kl_jitter * eye_block

        diag_x, factor_x = SegFormer_GSFVIGenerativeNetwork._batched_lowrank_residual_diag_decomposition(
            p_cov_x, rank=prior_rank, eps=eps
        )
        diag_y, factor_y = SegFormer_GSFVIGenerativeNetwork._batched_lowrank_residual_diag_decomposition(
            p_cov_y, rank=prior_rank, eps=eps
        )
        prior_diag = torch.cat((diag_x, diag_y), dim=1)
        prior_factor = SegFormer_GSFVIGenerativeNetwork._assemble_blockdiag_factor(factor_x, factor_y)

        posterior_rank = max(
            1,
            min(
                int(posterior_rank),
                posterior_spectral["x"]["u"].size(2),
                posterior_spectral["y"]["u"].size(2),
            ),
        )
        q_u_x = posterior_spectral["x"]["u"][:, :, :posterior_rank]
        q_lam_x = posterior_spectral["x"]["lambda"][:, :posterior_rank].clamp_min(eps)
        q_u_y = posterior_spectral["y"]["u"][:, :, :posterior_rank]
        q_lam_y = posterior_spectral["y"]["lambda"][:, :posterior_rank].clamp_min(eps)
        q_u, q_lambda = FVIBlockDiagonalPosterior._assemble_block_diagonal_spectral(
            q_u_x, q_lam_x, q_u_y, q_lam_y
        )

        p_inv_u = SegFormer_GSFVIGenerativeNetwork._woodbury_solve(prior_diag, prior_factor, q_u)
        gram_diag = (q_u * p_inv_u).sum(dim=1)
        trace_raw = (q_lambda * gram_diag).sum(dim=1)

        mean_diff = (p_mean - q_mean).unsqueeze(-1)
        p_inv_diff = SegFormer_GSFVIGenerativeNetwork._woodbury_solve(prior_diag, prior_factor, mean_diff)
        quad_raw = torch.matmul(mean_diff.transpose(1, 2), p_inv_diff).squeeze(-1).squeeze(-1)

        diag_logdet = torch.log(prior_diag.clamp_min(eps)).sum(dim=1)
        d_inv_factor = prior_diag.reciprocal().unsqueeze(-1) * prior_factor
        eye = torch.eye(prior_factor.size(2), device=prior_factor.device, dtype=prior_factor.dtype).unsqueeze(0)
        middle = eye + prior_factor.transpose(1, 2) @ d_inv_factor
        chol = torch.linalg.cholesky(middle)
        logdet_middle = 2.0 * torch.log(torch.diagonal(chol, dim1=1, dim2=2)).sum(dim=1)
        logdet_p = diag_logdet + logdet_middle
        logdet_q_pseudo = torch.log(q_lambda.clamp_min(eps)).sum(dim=1)

        logdet_term_raw = logdet_p - logdet_q_pseudo - dim
        kl = 0.5 * (trace_raw + quad_raw + logdet_term_raw)
        if not return_components:
            return kl
        return {
            "kl": kl,
            "logdet_term": 0.5 * logdet_term_raw,
            "trace_term": 0.5 * trace_raw,
            "quad_term": 0.5 * quad_raw,
            "logdet_p": logdet_p,
            "logdet_q": logdet_q_pseudo,
            "dim": dim,
            "prior_rank_per_block": int(prior_rank),
            "posterior_rank_per_block": int(posterior_rank),
        }


    def forward_reg(self, src, tgt, qalpha_mode=None, sample_upstream=None):
        resolved_qalpha_mode = self._resolve_qalpha_mode(qalpha_mode)
        resolved_sample_upstream = self._resolve_upstream_sampling(sample_upstream)
        palpha = self.encoder(src, tgt, qalpha_mode=resolved_qalpha_mode,
                              sample_upstream=resolved_sample_upstream)
        alpha_points = self._flatten_alpha(palpha)
        phi = self.decoder(alpha_points)
        w_src = self.transformer(src, phi)
        return phi, w_src, palpha

    def test(self, src, tgt, qalpha_mode=None, sample_upstream=None, mc_samples=None):
        mc_samples = self.test_mc_samples if mc_samples is None else int(mc_samples)
        if mc_samples <= 1:
            return self.forward_reg(src, tgt, qalpha_mode=qalpha_mode,
                sample_upstream=sample_upstream)
        _qalpha_mode = "sample"
        _sample_upstream = True
        phi_list = []
        palpha_list = []
        for _ in range(mc_samples):
            phi_i, _, palpha_i = self.forward_reg(src, tgt, qalpha_mode=_qalpha_mode,
                sample_upstream=_sample_upstream)
            phi_list.append(phi_i)
            palpha_list.append(palpha_i)

        phi = torch.stack(phi_list, dim=0).mean(dim=0)
        w_src = self.transformer(src, phi)
        palpha = torch.stack(palpha_list, dim=0).mean(dim=0)
        return phi, w_src, palpha

    def objective(self, src, tgt):
        prior_mean, prior_cov = self.build_prior(src, tgt)

        similarity_terms = []
        smooth_terms = []

        sample_upstream = self._resolve_upstream_sampling(None)
        qalpha_mode = self._resolve_qalpha_mode(None)
        qalpha_features = self.encoder.extract_qalpha_features(src, tgt, sample=sample_upstream)
        posterior = self.encoder.forward_posterior_moments_from_features(
            qalpha_features,
            cov_jitter=self.qalpha_linearization_jitter,
            kl_jitter=self.posterior_kl_jitter,
            spectral_rank=self.posterior_kl_lowrank_rank,
        )
        posterior["alpha_mean_points"] = self._flatten_alpha(posterior["alpha_mean_map"])
        posterior["alpha_mean_flat"] = self._flatten_alpha_xy(posterior["alpha_mean_map"])
        kl_components = self._batched_paper_lowrank_kl(
            q_mean=posterior["alpha_mean_flat"],
            posterior_spectral=posterior["alpha_cov_spectral"],
            p_mean=prior_mean,
            p_cov=prior_cov,
            prior_rank=self.prior_kl_lowrank_rank,
            posterior_rank=self.posterior_kl_lowrank_rank,
            kl_jitter=self.posterior_kl_jitter,
            return_components=True,
        )
        posterior_kl = kl_components["kl"]

        mc_samples = max(1, self.train_mc_samples)
        for _ in range(mc_samples):
            alpha_sample_map = self.encoder.forward_qalpha(qalpha_features, qalpha_mode=qalpha_mode)
            alpha_sample_points = self._flatten_alpha(alpha_sample_map)
            flow = self.decoder(alpha_sample_points)
            src_s = self.transformer(src, flow)

            similarity_terms.append(self.similarity_loss(src_s, tgt))
            smooth_terms.append(self.bending_energy(alpha_sample_points))

        similarity_loss = torch.stack(similarity_terms, dim=0).mean(dim=0)
        smooth_term = torch.stack(smooth_terms, dim=0).mean(dim=0)
        posterior_kl_logdet_term = kl_components["logdet_term"]
        posterior_kl_trace_term = kl_components["trace_term"]
        posterior_kl_quad_term = kl_components["quad_term"]
        total_loss = (
            similarity_loss
            + smooth_term / self.similarity_factor
            + self.posterior_kl_weight * posterior_kl
        )

        return {
            'loss': total_loss,
            'similarity_loss': similarity_loss,
            'smooth_term': smooth_term,
            'posterior_kl': posterior_kl,
            'logdet': posterior_kl_logdet_term,
            'trace': posterior_kl_trace_term,
            'quad': posterior_kl_quad_term,
        }


    def uncertainty(self, src, tgt, src_seg, K):
        d_list = []
        w_list = []

        was_training = self.encoder.training
        self.encoder.train(True)
        for _ in range(K):
            alpha_map = self.encoder(src, tgt, qalpha_mode="sample")
            alpha_points = self._flatten_alpha(alpha_map)
            phi_xy = self.decoder(alpha_points)
            w_src = self.transformer(src_seg, phi_xy, mode='nearest')
            phi_r = torch.norm(phi_xy, dim=1, keepdim=True)
            phi = torch.cat([phi_xy, phi_r], dim=1)
            d_list.append(phi)
            w_list.append(w_src)
        self.encoder.train(was_training)

        d = torch.stack(d_list, dim=1)
        d_expect = d.mean(dim=1)
        dd_expect = (d * d).mean(dim=1)
        var = dd_expect - d_expect * d_expect

        w = torch.stack(w_list, dim=1)
        w_lbl = w.squeeze(2).long()
        num_classes = max(2, int(w_lbl.max().detach().item()) + 1)
        w_oh = torch.nn.functional.one_hot(w_lbl, num_classes=num_classes)
        probs = w_oh.float().mean(dim=1)
        probs = probs.permute(0, 3, 1, 2).contiguous()
        return var, d_expect, probs
