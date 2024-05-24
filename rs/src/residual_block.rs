use tch::{nn, Tensor};

pub struct ResidualBlock {
    conv1: nn::utils::spectral_norm::SpectralNorm,
    conv2: nn::utils::spectral_norm::SpectralNorm,
    downsample: bool,
    downsample_layer: Option<nn::Conv2D>,
}

impl ResidualBlock {
    pub fn new(in_channels: i64, out_channels: i64, downsample: bool) -> Self {
        let conv1 = nn::utils::spectral_norm::spectral_norm(
            nn::conv2d(
                in_channels,
                out_channels,
                3,
                nn::ConvConfig {
                    padding: 1,
                    ..Default::default()
                },
            ),
            1,
        );
        let conv2 = nn::utils::spectral_norm::spectral_norm(
            nn::conv2d(
                out_channels,
                out_channels,
                3,
                nn::ConvConfig {
                    padding: 1,
                    ..Default::default()
                },
            ),
            1,
        );
        let downsample_layer = if downsample {
            Some(nn::conv2d(
                in_channels,
                out_channels,
                1,
                nn::ConvConfig {
                    stride: 2,
                    ..Default::default()
                },
            ))
        } else {
            None
        };
        ResidualBlock {
            conv1,
            conv2,
            downsample,
            downsample_layer,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let residual = x.shallow_clone();
        let x = self.conv1.forward(x);
        let x = x.leaky_relu(0.2);
        let x = self.conv2.forward(&x);
        let x = if self.downsample {
            nn::functional::avg_pool2d(&x, 2, nn::functional::AvgPoolFuncOptions::default())
        } else {
            x
        };
        if let Some(downsample_layer) = &self.downsample_layer {
            let residual = downsample_layer.forward(&residual);
            x + residual
        } else {
            x + residual
        }
    }
}
