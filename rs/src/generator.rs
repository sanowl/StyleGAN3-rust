use rand::distributions::{Distribution, Normal, Uniform};

use mapping_network::MappingNetwork;
use style_layer::StyleLayer;

#[derive(Debug)]
pub struct Generator<T> {
    mapping: MappingNetwork<T>,
    style_layers: Vec<StyleLayer<T>>,
    to_rgb: Conv2D<T>,
}

impl<T: Copy + Default + std::ops::Mul<Output = T> + std::ops::Add<Output = T>> Generator<T> {
    pub fn new(
        latent_dim: usize,
        hidden_dim: usize,
        output_channels: usize,
        num_layers: usize,
    ) -> Self {
        let mapping = MappingNetwork::new(latent_dim, hidden_dim, num_layers);

        let mut style_layers = Vec::with_capacity(num_layers + 1);

        for i in 0..num_layers {
            let layer = StyleLayer::new(hidden_dim, hidden_dim, hidden_dim, 3, i % 2 == 1);
            style_layers.push(layer);
        }

        // Final style layer (no upsampling, output_channels as output)
        let final_layer = StyleLayer::new(hidden_dim, hidden_dim, output_channels, 3, false);
        style_layers.push(final_layer);

        let to_rgb_weights = xavier_uniform(1, 1, output_channels, output_channels);
        let to_rgb_bias = vec![T::default(); output_channels];
        let to_rgb = Conv2D::new(to_rgb_weights, to_rgb_bias, (1, 1), (0, 0));

        Generator {
            mapping,
            style_layers,
            to_rgb,
        }
    }

    fn forward(&self, z: &[T]) -> Vec<Vec<Vec<T>>> {
        let w = self.mapping.forward(z);

        // Efficient repetition of w for each style layer
        let w = repeat_tensor(&w, self.style_layers.len());

        // Initialize x with proper dimensions (considering batch size)
        let mut x = xavier_uniform(4, 4, 1, self.style_layers[0].in_channels);

        for (i, style_layer) in self.style_layers.iter().enumerate() {
            let w_slice = &w[i];
            x = style_layer.forward(&x, w_slice);
        }

        let x = self.to_rgb.forward(&x);
        let x = apply_tanh(&x);

        x
    }
}

fn xavier_uniform<T: Copy + Default + std::ops::Mul<Output = T> + std::ops::Add<Output = T>>(
    height: usize,
    width: usize,
    in_channels: usize,
    out_channels: usize,
) -> Vec<Vec<Vec<Vec<T>>>> {
    let mut weights = Vec::with_capacity(out_channels);
    let range = Uniform::new(-1.0, 1.0);
    let scale = (6.0 / ((height * width * in_channels + out_channels) as f64)).sqrt();

    for _ in 0..out_channels {
        let mut channel = Vec::with_capacity(in_channels);
        for _ in 0..in_channels {
            let mut filter = Vec::with_capacity(height);
            for _ in 0..height {
                let mut row = Vec::with_capacity(width);
                for _ in 0..width {
                    row.push(range.sample(&mut rand::thread_rng()) * scale);
                }
                filter.push(row);
            }
            channel.push(filter);
        }
        weights.push(channel);
    }

    weights
}

fn repeat_tensor<T: Copy>(tensor: &[T], num_repeats: usize) -> Vec<Vec<T>> {
    let mut repeated = Vec::with_capacity(num_repeats);
    for _ in 0..num_repeats {
        repeated.push(tensor.to_vec());
    }
    repeated
}

fn apply_tanh<T: Copy + Default + std::ops::Mul<Output = T> + std::ops::Add<Output = T>>(
    tensor: &[Vec<Vec<T>>],
) -> Vec<Vec<Vec<T>>> {
    let mut output = Vec::with_capacity(tensor.len());
    for channel in tensor {
        let mut channel_output = Vec::with_capacity(channel.len());
        for row in channel {
            let mut row_output = Vec::with_capacity(row.len());
            for value in row {
                row_output.push(tanh(*value));
            }
            channel_output.push(row_output);
        }
        output.push(channel_output);
    }
    output
}

fn tanh<T: Copy + Default + std::ops::Mul<Output = T> + std::ops::Add<Output = T>>(x: T) -> T {
    let exp_2x = (2.0 * x as f64).exp();
    (exp_2x - 1.0) / (exp_2x + 1.0) as T
}

struct Conv2D<T> {
    weights: Vec<Vec<Vec<Vec<T>>>>,
    bias: Vec<T>,
    stride: (usize, usize),
    padding: (usize, usize),
}

impl<T: Copy + Default + std::ops::Mul<Output = T> + std::ops::Add<Output = T>> Conv2D<T> {
    fn new(
        weights: Vec<Vec<Vec<Vec<T>>>>,
        bias: Vec<T>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Self {
        Conv2D {
            weights,
            bias,
            stride,
            padding,
        }
    }

    pub fn forward(&self, x: &[Vec<Vec<T>>]) -> Vec<Vec<Vec<T>>> {
        let (batch_size, channels, height, width) =
            (x.len(), x[0].len(), x[0][0].len(), x[0][0][0].len());

        let query = self.query.forward(x);
        let query = view_and_permute(&query, batch_size, channels / 8, width * height);

        let key = self.key.forward(x);
        let key = view(&key, batch_size, channels / 8, width * height);

        let attention = batched_matrix_multiplication(&query, &key.transpose());
        let attention = softmax(&attention, -1);

        let value = self.value.forward(x);
        let value = view(&value, batch_size, channels, width * height);

        let out = batched_matrix_multiplication(&value, &attention.transpose());
        let out = view_and_reshape(&out, batch_size, channels, height, width);

        let gamma_tensor = repeat(&self.gamma, batch_size, channels, height, width);
        element_wise_addition(&element_wise_multiplication(&gamma_tensor, &out), x)
    }
}
