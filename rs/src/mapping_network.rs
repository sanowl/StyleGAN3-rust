use rand::distributions::{Distribution, Uniform};

pub struct MappingNetwork<T> {
    weights: Vec<Vec<Vec<T>>>,
    biases: Vec<Vec<T>>,
    num_layers: usize,
    hidden_dim: usize,
}

impl<T: Copy + Default + std::ops::Mul<Output = T> + std::ops::Add<Output = T>> MappingNetwork<T> {
    pub fn new(latent_dim: usize, hidden_dim: usize, num_layers: usize) -> Self {
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        let range = Uniform::new(-1.0, 1.0);

        // Initialize first layer weights and biases
        let mut first_layer_weights = Vec::with_capacity(hidden_dim);
        let mut first_layer_biases = Vec::with_capacity(hidden_dim);
        for _ in 0..hidden_dim {
            let mut neuron_weights = Vec::with_capacity(latent_dim);
            for _ in 0..latent_dim {
                neuron_weights.push(range.sample(&mut rand::thread_rng()));
            }
            first_layer_weights.push(neuron_weights);
            first_layer_biases.push(range.sample(&mut rand::thread_rng()));
        }
        weights.push(first_layer_weights);
        biases.push(first_layer_biases);

        // Initialize hidden layer weights and biases
        for _ in 0..num_layers - 1 {
            let mut layer_weights = Vec::with_capacity(hidden_dim);
            let mut layer_biases = Vec::with_capacity(hidden_dim);
            for _ in 0..hidden_dim {
                let mut neuron_weights = Vec::with_capacity(hidden_dim);
                for _ in 0..hidden_dim {
                    neuron_weights.push(range.sample(&mut rand::thread_rng()));
                }
                layer_weights.push(neuron_weights);
                layer_biases.push(range.sample(&mut rand::thread_rng()));
            }
            weights.push(layer_weights);
            biases.push(layer_biases);
        }

        MappingNetwork {
            weights,
            biases,
            num_layers,
            hidden_dim,
        }
    }

    pub fn forward(&self, x: &[T]) -> Vec<T> {
        let mut activations = x.to_vec();

        for layer in 0..self.num_layers {
            let mut new_activations = Vec::with_capacity(self.hidden_dim);
            for neuron in 0..self.hidden_dim {
                let mut weighted_sum = self.biases[layer][neuron];
                for (weight, input) in self.weights[layer][neuron].iter().zip(activations.iter()) {
                    weighted_sum = weighted_sum + weight * input;
                }
                new_activations.push(relu(weighted_sum));
            }
            activations = new_activations;
        }

        activations
    }
}

fn relu<T: Copy + Default + std::ops::Mul<Output = T> + std::ops::Add<Output = T>>(x: T) -> T {
    if x < T::default() {
        T::default()
    } else {
        x
    }
}
