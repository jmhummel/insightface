import torch
import numpy as np

class ResidualBlock(torch.nn.Module):
    def __init__(self, ch, norm=None):
        super().__init__()

        if norm == "batchnorm":
            self.norm1 = torch.nn.BatchNorm2d(ch)
            self.norm2 = torch.nn.BatchNorm2d(ch)
            conv_bias = False
        elif norm == "groupnorm":
            self.norm1 = torch.nn.GroupNorm(min(ch, 32), ch)
            self.norm2 = torch.nn.GroupNorm(min(ch, 32), ch)
            conv_bias = False
        elif not norm:
            self.norm1 = None
            self.norm2 = None
            conv_bias = True
        else:
            raise ValueError(f"Unsupported norm type {norm}")

        self.conv1 = torch.nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=conv_bias)
        init_conv(self.conv1)
        self.conv2 = torch.nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=conv_bias)
        init_conv(self.conv2)
        self.act = torch.nn.LeakyReLU(0.2)

    def forward(self, inp):
        x = self.conv1(inp)
        if self.norm1:
            x = self.norm1(x)
        x = self.act(x)

        x = self.conv2(x)
        if self.norm2:
            x = self.norm2(x)
        x = self.act(x + inp)

        return x

def convolution_aware(tensor: torch.Tensor, eps_std=0.05) -> torch.Tensor:
    n_out_channels, n_in_channels, row, column = tensor.shape

    fan_in = n_in_channels * (row * column)

    kernel_shape = (row, column)
    kernel_fft_shape = np.fft.rfft2(np.zeros(kernel_shape)).shape

    basis_size = np.prod(kernel_fft_shape)
    if basis_size == 1:
        x = np.random.normal( 0.0, eps_std, (n_out_channels, n_in_channels, basis_size) )
    else:
        nbb = n_in_channels // basis_size + 1
        x = np.random.normal(0.0, 1.0, (n_out_channels, nbb, basis_size, basis_size))
        x = x + np.transpose(x, (0,1,3,2) ) * (1-np.eye(basis_size))
        u, _, v = np.linalg.svd(x)
        x = np.transpose(u, (0,1,3,2) )
        x = np.reshape(x, (n_out_channels, -1, basis_size) )
        x = x[:,:n_in_channels,:]

    x = np.reshape(x, ( (n_out_channels,n_in_channels,) + kernel_fft_shape ) )

    x = np.fft.irfft2( x, kernel_shape ) \
        + np.random.normal(0, eps_std, (n_out_channels,n_in_channels,)+kernel_shape)

    x = x * np.sqrt( (2/fan_in) / np.var(x) )

    with torch.no_grad():
        tensor.data = torch.tensor(x.astype(np.float32))

    return tensor

def init_conv(mod):
    """Helper to init the weight and bias of a conv according to dfl"""
    convolution_aware(mod.weight)
    torch.nn.init.zeros_(mod.bias)

class DownscaleBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, norm=None):
        super(DownscaleBlock, self).__init__()

        if norm == "batchnorm":
            self.norm = torch.nn.BatchNorm2d(out_channels)
            conv_bias = False
        elif norm == "groupnorm":
            self.norm = torch.nn.GroupNorm(min(out_channels, 32), out_channels)
            conv_bias = False
        elif not norm:
            self.norm = None
            conv_bias = True
        else:
            raise ValueError(f"Unsupported norm type {norm}")


        self.conv = torch.nn.Conv2d(in_channels,
                                    out_channels,
                                    kernel_size=5,
                                    stride=2,
                                    padding=2,
                                    bias=conv_bias,
                                    )
        init_conv(self.conv)
        self.act = torch.nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        x = self.act(x)
        return x

class Encoder(torch.nn.Module):
    """Encoder architecture form DFL.
    Default parameters is an SAEHD architecture.
    """
    def __init__(self,
                 in_channels,
                 num_channels,
                 num_downscale=4,
                 max_channel_multiplier=8,
                 initial_res=False,
                 norm=None,
                 ):
        super(Encoder, self).__init__()

        self.num_downscales = num_downscale
        self.downscales = torch.nn.ModuleList()
        self.initial_res = initial_res

        if initial_res:
            self.initial_res = ResidualBlock(num_channels, norm=norm)
        for down in range(self.num_downscales):
            out_channels = num_channels * (min(2**down, max_channel_multiplier))
            block = DownscaleBlock(in_channels, out_channels, norm=norm)
            self.downscales.append(block)
            in_channels = out_channels

        self.out_channels = out_channels

    def forward(self, x):
        for idx, down in enumerate(self.downscales):
            x = down(x)
            if idx == 0 and self.initial_res:
                x = self.initial_res(x)

        return x


def dense_norm(x, eps=1e-6):
    scale = torch.mean(x.square(), dim=-1, keepdim=True) + eps
    return x / scale.sqrt()

class AmpEncoder(torch.nn.Module):
    """AMP style Encoder
    An encoder with some extra res blocks before and after and a linear projection down to a latent.
    """

    def __init__(self,
                 in_channels,
                 encoder_channels,
                 ae_channels,
                 resolution,
                 num_downscale=5,
                 max_channel_multiplier=8,
                 norm=None,
                 ):

        super().__init__()

        self.resolution = resolution
        spatial_output_size = resolution // (2 ** num_downscale)
        self.out_channels = ae_channels
        self.encoder = Encoder(in_channels,
                               encoder_channels,
                               num_downscale=num_downscale,
                               max_channel_multiplier=max_channel_multiplier,
                               initial_res=True,
                               norm=norm,
                               )

        self.final_res = ResidualBlock(self.encoder.out_channels, norm=norm)
        feature_map_size = spatial_output_size*spatial_output_size*self.encoder.out_channels
        self.flatten = torch.nn.Flatten()
        self.norm = dense_norm # pixel_norm in dfl is just a repeat of dense_norm
        self.dense1 = torch.nn.Linear(feature_map_size, ae_channels)
        # dfl init
        torch.nn.init.xavier_uniform_(self.dense1.weight)
        torch.nn.init.zeros_(self.dense1.bias)

    def forward(self, x):
        x = self.encoder(x)
        x = self.final_res(x)
        x = self.flatten(x)
        x = self.norm(x)
        x = self.dense1(x)
        return x

def get_amp(num_classes):
    return AmpEncoder(3, 128, num_classes, 112)