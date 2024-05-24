use tch::{nn, Tensor};

pub struct Blur {
    kernel: Tensor,
}

impl Blur {
    pub fn new(channels: i64) -> Self {
        let kernel = Tensor::of_slice(&[1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0])
            .view((1, 1, 3, 3))
            .repeat(&[channels, 1, 1, 1])
            / 16.0;
        Blur { kernel }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        nn::functional::conv2d(
            x,
            &self.kernel,
            nn::ConvConfig {
                stride: 1,
                padding: 1,
                groups: x.size()[1],
                ..Default::default()
            },
        )
    }
}
