use tch::{nn, Tensor};

pub struct NoiseInjection {
    scale: Tensor,
}
impl NoiseInjection {
    pub fn new() -> Self {
        let scale = Tensor::zeros(&[1], tch::kind::FLOAT_CPU);
        NoiseInjection { scale }
    }
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let noise = Tensor::randn_like(x);
        x + &self.scale * noise
    }
}
