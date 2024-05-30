use tch::{nn, Tensor};

pub struct MappingNetwork {
    mapping: nn::Sequential,
}

impl MappingNetwork {
    pub fn new(latent_dim: i64, hidden_dim: i64, num_layers: i64) -> Self {
        let mut layers = nn::sequential();
        layers.add(nn::linear(latent_dim, hidden_dim, Default::default()));
        layers.add_fn(|xs| xs.relu());
        for _ in 0..num_layers - 1 {
            layers.add(nn::linear(hidden_dim, hidden_dim, Default::default()));
            layers.add_fn(|xs| xs.relu());
        }
        MappingNetwork { mapping: layers }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        self.mapping.forward(x)
    }
}


