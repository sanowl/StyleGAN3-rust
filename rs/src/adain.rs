use rand::distributions::{Distribution, Normal, Uniform};

pub struct AdaIN<T> {
    norm: InstanceNorm2D<T>,
    style: Linear<T>,
}

impl<T: Copy + Default + std::ops::Mul<Output = T> + std::ops::Add<Output = T>> AdaIN<T> {
    pub fn new(latent_dim: usize, channels: usize) -> Self {
        let norm = InstanceNorm2D::new(channels);
        let style = Linear::new(latent_dim, channels * 2);

        AdaIN { norm, style }
    }

    pub fn forward(&self, x: &[Vec<Vec<T>>], w: &[T]) -> Vec<Vec<Vec<T>>> {
        let style = self.style.forward(w);
        let (gamma, beta) = style.chunks(2).collect::<Vec<_>>();

        let mut output = self.norm.forward(&x);
        for c in 0..output.len() {
            for h in 0..output[c].len() {
                for w in 0..output[c][h].len() {
                    output[c][h][w] = (1.0 + gamma[c / 2]) * output[c][h][w] + beta[c / 2];
                }
            }
        }

        output
    }
}

struct InstanceNorm2D<T> {
    gamma: Vec<T>,
    beta: Vec<T>,
    epsilon: T,
}

impl<T: Copy + Default + std::ops::Mul<Output = T> + std::ops::Add<Output = T>> InstanceNorm2D<T> {
    fn new(channels: usize) -> Self {
        let gamma = vec![T::from(1.0).unwrap(); channels];
        let beta = vec![T::default(); channels];
        let epsilon = T::from(1e-5).unwrap();

        InstanceNorm2D {
            gamma,
            beta,
            epsilon,
        }
    }

    fn forward(&self, input: &[Vec<Vec<T>>]) -> Vec<Vec<Vec<T>>> {
        let (channels, height, width) = (input.len(), input[0].len(), input[0][0].len());
        let mut output = vec![vec![vec![T::default(); width]; height]; channels];

        for c in 0..channels {
            let (mu, sigma_squared) = calculate_instance_statistics(&input[c]);
            for h in 0..height {
                for w in 0..width {
                    output[c][h][w] = (input[c][h][w] - mu) / (sigma_squared + self.epsilon).sqrt();
                    output[c][h][w] = self.gamma[c] * output[c][h][w] + self.beta[c];
                }
            }
        }

        output
    }
}

fn calculate_instance_statistics<
    T: Copy + Default + std::ops::Mul<Output = T> + std::ops::Add<Output = T>,
>(
    input: &[Vec<T>],
) -> (T, T) {
    let mut mu = T::default();
    let mut sigma_squared = T::default();
    let n = (input.len() * input[0].len()) as f64;

    for h in 0..input.len() {
        for w in 0..input[h].len() {
            mu = mu + input[h][w];
        }
    }
    mu = mu / T::from(n).unwrap();

    for h in 0..input.len() {
        for w in 0..input[h].len() {
            sigma_squared = sigma_squared + (input[h][w] - mu).powi(2);
        }
    }
    sigma_squared = sigma_squared / T::from(n).unwrap();

    (mu, sigma_squared)
}

struct Linear<T> {
    weights: Vec<Vec<T>>,
    bias: Vec<T>,
}

impl<T: Copy + Default + std::ops::Mul<Output = T> + std::ops::Add<Output = T>> Linear<T> {
    fn new(input_size: usize, output_size: usize) -> Self {
        let range = Uniform::new(-1.0, 1.0);
        let mut weights = Vec::with_capacity(output_size);
        let mut bias = Vec::with_capacity(output_size);

        for _ in 0..output_size {
            let mut neuron_weights = Vec::with_capacity(input_size);
            for _ in 0..input_size {
                neuron_weights.push(range.sample(&mut rand::thread_rng()));
            }
            weights.push(neuron_weights);
            bias.push(T::default());
        }

        Linear { weights, bias }
    }

    fn forward(&self, input: &[T]) -> Vec<T> {
        let mut output = Vec::with_capacity(self.weights.len());

        for neuron in &self.weights {
            let mut weighted_sum = self.bias[output.len()];
            for (weight, input) in neuron.iter().zip(input.iter()) {
                weighted_sum = weighted_sum + weight * input;
            }
            output.push(weighted_sum);
        }

        output
    }
}

trait Chunks<T> {
    fn chunks(&self, chunk_size: usize) -> Vec<Vec<T>>;
}

impl<T: Copy> Chunks<T> for Vec<T> {
    fn chunks(&self, chunk_size: usize) -> Vec<Vec<T>> {
        let mut chunks = Vec::new();
        let mut chunk = Vec::with_capacity(chunk_size);

        for (i, value) in self.iter().enumerate() {
            if i % chunk_size == 0 && i != 0 {
                chunks.push(chunk);
                chunk = Vec::with_capacity(chunk_size);
            }
            chunk.push(*value);
        }

        if !chunk.is_empty() {
            chunks.push(chunk);
        }

        chunks
    }
}
