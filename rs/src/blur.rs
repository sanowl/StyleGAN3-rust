use std::error::Error;
use tch::{nn, Tensor};

pub struct Blur {
    kernel: Tensor,
}

impl Blur {
    pub fn new(channels: i64) -> Result<Self, Box<dyn Error>> {
        if channels <= 0 {
            return Err("Channels must be a positive integer".into());
        }

        let kernel = Tensor::of_slice(&[1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0])
            .view((1, 1, 3, 3))
            .repeat(&[channels, 1, 1, 1])
            / 16.0;

        Ok(Blur { kernel })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor, Box<dyn Error>> {
        if x.size().len() != 4 {
            return Err("Input tensor must have 4 dimensions".into());
        }

        if x.size()[1] != self.kernel.size()[1] {
            return Err("Input tensor must have the same number of channels as the kernel".into());
        }

        Ok(nn::conv2d(x, &self.kernel, &[1, 1], &[1, 1], &[1, 1]))
    }
}
