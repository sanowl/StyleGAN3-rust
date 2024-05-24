use tch::{nn, nn::Module, Tensor};

pub struct AdaIN {
    norm: nn::InstanceNorm2D,
    style: nn::Linear,
}

impl AdaIN {
    pub fn new<'a>(vs: &nn::Path<'a>, latent_dim: i64, channels: i64) -> Self {
        let norm = nn::instance_norm2d(vs / "norm", channels, Default::default());
        let style = nn::linear(vs / "style", latent_dim, channels * 2, Default::default());
        AdaIN { norm, style }
    }

    pub fn forward(&self, x: &Tensor, w: &Tensor) -> Tensor {
        let style = self.style.forward(w).unsqueeze(-1).unsqueeze(-1);
        let (gamma, beta) = style.chunk(2, 1);
        (1 + gamma) * self.norm.forward(x) + beta
    }
}
