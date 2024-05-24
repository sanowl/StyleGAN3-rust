use tch::{nn, Tensor};

use crate::mapping_network::MappingNetwork;
use crate::style_layer::StyleLayer;

pub struct Generator {
    mapping: MappingNetwork,
    style_layers: Vec<StyleLayer>,
    to_rgb: nn::Sequential,
}

impl Generator {
    pub fn new(latent_dim: i64, hidden_dim: i64, output_channels: i64, num_layers: i64) -> Self {
        let mapping = MappingNetwork::new(latent_dim, hidden_dim, num_layers);
        let style_layers = vec![
            StyleLayer::new(hidden_dim, hidden_dim, hidden_dim, 3, true, false),
            StyleLayer::new(hidden_dim, hidden_dim, hidden_dim, 3, true, true),
            StyleLayer::new(hidden_dim, hidden_dim, hidden_dim, 3, true, false),
            StyleLayer::new(hidden_dim, hidden_dim, hidden_dim, 3, true, false),
            StyleLayer::new(hidden_dim, hidden_dim, hidden_dim, 3, true, false),
            StyleLayer::new(hidden_dim, hidden_dim, hidden_dim, 3, true, true),
            StyleLayer::new(hidden_dim, hidden_dim, hidden_dim, 3, true, false),
            StyleLayer::new(hidden_dim, hidden_dim, hidden_dim, 3, true, false),
            StyleLayer::new(hidden_dim, hidden_dim, output_channels, 3, false, false),
        ];
        let to_rgb = nn::seq()
            .add(nn::conv2d(
                output_channels,
                output_channels,
                1,
                Default::default(),
            ))
            .add_fn(|xs| xs.tanh());
        Generator {
            mapping,
            style_layers,
            to_rgb,
        }
    }

    pub fn forward(&self, z: &Tensor) -> Tensor {
        let w = self.mapping.forward(z);
        let w = w
            .unsqueeze(1)
            .repeat(&[1, self.style_layers.len() as i64, 1]);
        let mut x = Tensor::randn(
            &[z.size()[0], self.style_layers[0].conv.ws.size()[0], 4, 4],
            (tch::Kind::Float, z.device()),
        );
        for (i, style_layer) in self.style_layers.iter().enumerate() {
            let w_slice = w.select(1, i as i64);
            x = style_layer.forward(&x, &w_slice);
        }
        self.to_rgb.forward(&x)
    }
}
