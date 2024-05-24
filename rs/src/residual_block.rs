use rayon::prelude::*;
use std::fmt;

#[derive(Debug)]
pub enum ResidualBlockError {
    Conv2DError(Conv2DError),
    InvalidDimensions(String),
}

impl fmt::Display for ResidualBlockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ResidualBlockError::Conv2DError(err) => write!(f, "Conv2D error: {}", err),
            ResidualBlockError::InvalidDimensions(msg) => write!(f, "Invalid dimensions: {}", msg),
        }
    }
}

impl std::error::Error for ResidualBlockError {}

pub struct ResidualBlock<T> {
    conv1: Conv2D<T>,
    conv2: Conv2D<T>,
    downsample: bool,
    downsample_layer: Option<Conv2D<T>>,
}

impl<T: Copy + Default + std::ops::Mul<Output = T> + std::ops::Add<Output = T> + Send + Sync>
    ResidualBlock<T>
{
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        downsample: bool,
    ) -> Result<Self, ResidualBlockError> {
        let conv1_weights = xavier_uniform(3, 3, in_channels, out_channels);
        let conv1_bias = vec![T::default(); out_channels];
        let conv1 = Conv2D::new(conv1_weights, conv1_bias, (1, 1), (1, 1))?;

        let conv2_weights = xavier_uniform(3, 3, out_channels, out_channels);
        let conv2_bias = vec![T::default(); out_channels];
        let conv2 = Conv2D::new(conv2_weights, conv2_bias, (1, 1), (1, 1))?;

        let downsample_layer = if downsample {
            let downsample_weights = xavier_uniform(1, 1, in_channels, out_channels);
            let downsample_bias = vec![T::default(); out_channels];
            Some(Conv2D::new(
                downsample_weights,
                downsample_bias,
                (2, 2),
                (0, 0),
            )?)
        } else {
            None
        };

        Ok(ResidualBlock {
            conv1,
            conv2,
            downsample,
            downsample_layer,
        })
    }

    pub fn forward(&self, x: &[Vec<Vec<T>>]) -> Result<Vec<Vec<Vec<T>>>, ResidualBlockError> {
        let residual = x.to_vec();

        let x = self
            .conv1
            .forward_parallel(x)
            .map_err(|err| ResidualBlockError::Conv2DError(err))?;
        let x = leaky_relu_parallel(&x, 0.2);

        let x = self
            .conv2
            .forward_parallel(&x)
            .map_err(|err| ResidualBlockError::Conv2DError(err))?;

        let x = if self.downsample {
            avg_pool2d_parallel(&x, 2, 2)
        } else {
            x
        };

        let output = if let Some(downsample_layer) = &self.downsample_layer {
            let residual = downsample_layer
                .forward_parallel(&residual)
                .map_err(|err| ResidualBlockError::Conv2DError(err))?;

            if residual.len() != x.len()
                || residual[0].len() != x[0].len()
                || residual[0][0].len() != x[0][0].len()
            {
                return Err(ResidualBlockError::InvalidDimensions(
                    "Residual and output dimensions do not match".to_string(),
                ));
            }

            element_wise_addition_parallel(&x, &residual)
        } else {
            element_wise_addition_parallel(&x, &residual)
        };

        Ok(output)
    }
}

fn avg_pool2d_parallel<
    T: Copy + Default + std::ops::Div<Output = T> + std::ops::Add<Output = T> + Send + Sync,
>(
    input: &[Vec<Vec<T>>],
    kernel_size: usize,
    stride: usize,
) -> Vec<Vec<Vec<T>>> {
    let (input_channels, input_height, input_width) =
        (input.len(), input[0].len(), input[0][0].len());

    let output_height = (input_height - kernel_size) / stride + 1;
    let output_width = (input_width - kernel_size) / stride + 1;

    let mut output = vec![vec![vec![T::default(); output_width]; output_height]; input_channels];

    output.par_iter_mut().enumerate().for_each(|(c, channel)| {
        for oh in 0..output_height {
            for ow in 0..output_width {
                let mut sum = T::default();
                let mut count = 0;
                for kh in 0..kernel_size {
                    for kw in 0..kernel_size {
                        let input_h = oh * stride + kh;
                        let input_w = ow * stride + kw;
                        if input_h < input_height && input_w < input_width {
                            sum = sum + input[c][input_h][input_w];
                            count += 1;
                        }
                    }
                }
                channel[oh][ow] = sum / (count as T);
            }
        }
    });

    output
}

fn leaky_relu_parallel<
    T: Copy + Default + std::ops::Mul<Output = T> + std::ops::Add<Output = T> + Send + Sync,
>(
    input: &[Vec<Vec<T>>],
    negative_slope: f64,
) -> Vec<Vec<Vec<T>>> {
    let mut output = input.to_vec();
    output.par_iter_mut().for_each(|channel| {
        channel.par_iter_mut().for_each(|row| {
            row.par_iter_mut().for_each(|value| {
                if *value < T::default() {
                    *value = (*value as f64 * negative_slope) as T;
                }
            });
        });
    });
    output
}

fn element_wise_addition_parallel<T: Copy + Default + std::ops::Add<Output = T> + Send + Sync>(
    a: &[Vec<Vec<T>>],
    b: &[Vec<Vec<T>>],
) -> Vec<Vec<Vec<T>>> {
    let mut result = Vec::with_capacity(a.len());
    result.par_extend(a.par_iter().zip(b.par_iter()).map(|(batch_a, batch_b)| {
        let mut batch_result = Vec::with_capacity(batch_a.len());
        batch_result.par_extend(batch_a.par_iter().zip(batch_b.par_iter()).map(
            |(channel_a, channel_b)| {
                let mut channel_result = Vec::with_capacity(channel_a.len());
                channel_result.par_extend(channel_a.par_iter().zip(channel_b.par_iter()).map(
                    |(row_a, row_b)| {
                        row_a
                            .par_iter()
                            .zip(row_b.par_iter())
                            .map(|(a, b)| a + b)
                            .collect()
                    },
                ));
                channel_result
            },
        ));
        batch_result
    }));
    result
}
