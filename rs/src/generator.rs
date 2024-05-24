// generator.rs

use tch::{nn, nn::Module, Kind, Tensor};

use crate::mapping_network::MappingNetwork;
use crate::style_layer::StyleLayer;

#[derive(Debug)]
pub struct Generator {
    mapping: MappingNetwork,
    style_layers: Vec<StyleLayer>,
    to_rgb: nn::Sequential,
}

impl Generator {
    pub fn new(
        vs: &nn::Path,
        latent_dim: i64,
        hidden_dim: i64,
        output_channels: i64,
        num_layers: i64,
    ) -> Self {
        let mapping = MappingNetwork::new(vs / "mapping", latent_dim, hidden_dim, num_layers);
        let mut style_layers = Vec::new();

        for i in 0..num_layers {
            let layer = StyleLayer::new(
                vs / format!("style_layer_{}", i),
                hidden_dim,
                hidden_dim,
                hidden_dim,
                3,
                true,
                i % 2 == 1, // upsample for odd layers
            );
            style_layers.push(layer);
        }

        // Final style layer (no upsampling, output_channels as output)
        style_layers.push(StyleLayer::new(
            vs / format!("style_layer_{}", num_layers),
            hidden_dim,
            hidden_dim,
            output_channels,
            3,
            false,
            false,
        ));

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
}

impl Module for Generator {
    fn forward(&self, z: &Tensor) -> Tensor {
        let w = self.mapping.forward(z);

        // Efficient repetition of w for each style layer
        let w = w.unsqueeze(1).expand(&[-1, self.style_layers.len() as i64, -1]);

        // Initialize x with proper dimensions (considering batch size)
        let mut x = Tensor::randn(
            &[z.size()[0], self.style_layers[0].in_channels(), 4, 4],
            (Kind::Float, z.device()),
        );

        for (i, style_layer) in self.style_layers.iter().enumerate() {
            let w_slice = w.select(1, i as i64);
            x = style_layer.forward(&x, &w_slice);
        }

        self.to_rgb.forward(&x)
    }
}