use tch::{nn, Tensor};

pub struct SelfAttention {
    query: nn::Conv2D,
    key: nn::Conv2D,
    value: nn::Conv2D,
    gamma: Tensor,
}

impl SelfAttention {
    pub fn new(in_channels: i64) -> Self {
        let query = nn::conv2d(in_channels, in_channels / 8, 1, Default::default());
        let key = nn::conv2d(in_channels, in_channels / 8, 1, Default::default());
        let value = nn::conv2d(in_channels, in_channels, 1, Default::default());
        let gamma = Tensor::zeros(&[1], tch::kind::FLOAT_CPU);
        SelfAttention {
            query,
            key,
            value,
            gamma,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let (batch_size, channels, height, width) = x.size4().unwrap();
        let query = self
            .query
            .forward(x)
            .view((batch_size, -1, width * height))
            .permute(&[0, 2, 1]);
        let key = self.key.forward(x).view((batch_size, -1, width * height));
        let attention = Tensor::bmm(&query, &key);
        let attention = attention.softmax(-1, tch::Kind::Float);
        let value = self.value.forward(x).view((batch_size, -1, width * height));
        let out = Tensor::bmm(&value, &attention.permute(&[0, 2, 1]));
        let out = out.view((batch_size, channels, height, width));
        &self.gamma * out + x
    }
}
