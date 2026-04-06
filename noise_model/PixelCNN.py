import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from noise_model.GMM import GMM, get_gaussian_params, sampleFromMix


class ShiftedConvolution(nn.Module):
    """Implements a 1D convolution with a shifted kernel.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Length of the convolutional kernel.
    dilation : int
        Dilation factor.
    first : bool
        Whether this is the first convolution in the network.
        
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, first=False):
        super().__init__()

        shift = dilation * (kernel_size - 1)
        self.pad = nn.ConstantPad1d((shift, 0), 0)

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

        self.first = first
        if self.first:
            mask = torch.ones(kernel_size)
            mask[-1] = 0
            self.register_buffer("mask", mask[None, None])

    def forward(self, x):
        x = self.pad(x)
        if self.first:
            self.conv.weight.data *= self.mask
        x = self.conv(x)
        return x


class GatedBlock(nn.Module):
    """A gated activation unit.


    Parameters
    ----------
    n_filters : int
        Number of hidden channels.
    **kwargs
        Additional arguments for the convolutions.

    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, first=False):
        super().__init__()

        self.in_conv = ShiftedConvolution(
            in_channels, 2 * out_channels, kernel_size, dilation, first
        )
        self.out_conv = nn.Conv1d(out_channels, out_channels, 1)
        if in_channels == out_channels:
            self.do_skip = True
        else:
            self.do_skip = False

    def forward(self, x):
        feat = self.in_conv(x)
        tan, sig = torch.chunk(feat, 2, dim=1)
        out = torch.tanh(tan) * torch.sigmoid(sig)
        out = self.out_conv(out)
        if self.do_skip:
            out = out + x
        return out


class PixelCNN(GMM):
    """A CNN with attention gates and autoregressive convolutions

    Parameters
    ----------
    in_channels : int, optional
        The number of input channels. The default is 1.
    n_filters : int, optional
        The number of hidden channels. The default is 128.
    kernel_size : int, optional
        Side length of the convolutional kernel. The default is 5.
    n_gaussians : int, optional
        Number of components in the Gaussian mixture model. The default is 10.
    noise_mean : Float, optional
        Mean of the noise samples, used for normalisation of the data. The default is 0.
    noise_std : Float, optional
        Standard deviation of the noise samples, used for normalisation of the data. The default is 1.

    """

    def __init__(
        self,
        in_channels=1,
        n_filters=8,
        kernel_size=11,
        n_gaussians=2,
        noise_mean=0,
        noise_std=1,
        lr=2e-3,
    ):
        self.save_hyperparameters()
        super().__init__(n_gaussians, noise_mean, noise_std, lr)

        out_channels = n_gaussians * 3

        self.in_channels = in_channels

        self.gatedconvs = nn.Sequential(
            GatedBlock(in_channels, n_filters, kernel_size, first=True),
            GatedBlock(n_filters, n_filters, kernel_size, dilation=2),
            GatedBlock(n_filters, n_filters, kernel_size),
            GatedBlock(n_filters, n_filters, kernel_size, dilation=4),
            GatedBlock(n_filters, n_filters, kernel_size),
            GatedBlock(n_filters, n_filters, kernel_size, dilation=2),
            GatedBlock(n_filters, out_channels, kernel_size),
        )

    def _prepare_input(self, x):
        """Normalise shape to [B, C, W].

        Supported input shapes:
        - [B, 1, W] (legacy)
        - [B, 1, W, C] where C is 1 + d_model (PE features)
        """
        if x.dim() == 3:
            return x
        if x.dim() == 4:
            # [B, 1, W, C] -> [B, C, W], convolve only along W.
            return x[:, 0].moveaxis(-1, 1)
        raise ValueError(
            f"PixelCNN expects 3D or 4D input, got shape {tuple(x.shape)}."
        )

    def _target_channel(self, x):
        """Return modelled noise channel as [B, 1, W]."""
        if x.dim() == 3:
            return x[:, :1]
        if x.dim() == 4:
            # Use the first feature channel as the scalar noise target.
            return x[:, 0, :, :1].moveaxis(-1, 1)
        raise ValueError(
            f"PixelCNN expects 3D or 4D input, got shape {tuple(x.shape)}."
        )

    def forward(self, x):
        x = self._prepare_input(x)
        return self.gatedconvs(x)

    def loglikelihood(self, x):
        x_target = self._target_channel(x)
        x_target = (x_target - self.noise_mean) / self.noise_std
        x_cond = self._prepare_input(x)
        x_cond = (x_cond - self.noise_mean) / self.noise_std

        params = self.forward(x_cond)
        weights, means, stds = get_gaussian_params(params)

        loglikelihoods = Normal(means, stds).log_prob(x_target)
        temp = loglikelihoods.max(dim=1, keepdim=True)[0]
        loglikelihoods = loglikelihoods - temp
        loglikelihoods = loglikelihoods.exp()
        loglikelihoods = loglikelihoods * weights
        loglikelihoods = loglikelihoods.sum(dim=1, keepdim=True)
        loglikelihoods = loglikelihoods.log()
        loglikelihoods = loglikelihoods + temp
        return loglikelihoods

    @torch.no_grad()
    def sample(self, arr_shape, pe_features=None):
        """Sample noise autoregressively.

        Parameters
        ----------
        arr_shape : list
            [N, W] — number of samples and width.
        pe_features : torch.Tensor, optional
            Positional-encoding features with shape [N, W, d_model].
            When provided, sampling is conditioned on these features.

        Returns
        -------
        torch.Tensor
            Generated noise with shape [N, 1, W].
        """
        N, W = arr_shape[0], arr_shape[1]

        if pe_features is not None:
            d = pe_features.shape[-1]
            # 4-D array [N, 1, W, 1+d_model] kept in *normalised* space
            arr = torch.zeros((N, 1, W, 1 + d), dtype=torch.float).to(self.device)
            # Fill PE channels (normalised to match training)
            arr[:, 0, :, 1:] = (
                pe_features[:N].to(self.device) - self.noise_mean
            ) / self.noise_std

            for w in range(W):
                params = self.forward(arr[..., : w + 1, :])
                weights, means, stds = get_gaussian_params(params)
                samp = sampleFromMix(weights, means, stds)
                arr[:, 0, w, 0] = samp[..., w].squeeze()

            # Return de-normalised noise channel [N, 1, W]
            return arr[:, :, :, 0] * self.noise_std + self.noise_mean
        else:
            # Legacy 3-D sampling  [N, 1, W]
            arr = torch.zeros((N, 1, W), dtype=torch.float).to(self.device)
            for w in range(arr.shape[2]):
                params = self.forward(arr[..., : w + 1])
                weights, means, stds = get_gaussian_params(params)
                samp = sampleFromMix(weights, means, stds)
                arr[..., w] = samp[..., w]

            return arr * self.noise_std + self.noise_mean
