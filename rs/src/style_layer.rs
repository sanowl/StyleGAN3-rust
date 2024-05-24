use std::ops::Deref;
use std::error::Error;
use std::fmt;

struct AdaINError(String);

impl Error for AdaINError {
    fn description(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for AdaINError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl fmt::Debug for AdaINError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}



pub struct StyleLayer<T> {
    noise_injection: NoiseInjection,
    adain: AdaIN,
    conv: Conv2D<T>,
    upsample: bool,
    blur: Option<Blur>,
    attention: Option<SelfAttention>,
}

impl<T> Deref for StyleLayer<T> {
    type Target = Conv2D<T>;

    fn deref(&self) -> &Self::Target {
        &self.conv
    }
}

struct Conv2D<T> {
    weights: Vec<Vec<Vec<Vec<T>>>>,
    stride: (usize, usize),
    padding: (usize, usize),
}

impl<T: Copy + Default> Conv2D<T> {
    fn new(
        weights: Vec<Vec<Vec<Vec<T>>>>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Self {
        Conv2D {
            weights,
            stride,
            padding,
        }
    }

    fn forward(&self, input: &[Vec<Vec<T>>]) -> Vec<Vec<Vec<Vec<T>>>> {
        let mut output = Vec::new();
        for i in 0..(input.len() - self.weights.len() + 1) {
            let mut row = Vec::new();
            for j in 0..(input[0].len() - self.weights[0].len() + 1) {
                let mut channel = Vec::new();
                for k in 0..self.weights[0][0][0].len() {
                    let mut values = Vec::new();
                    for l in 0..self.weights[0][0].len() {
                        let mut value = T::default();
                        for m in 0..self.weights.len() {
                            for n in 0..self.weights[0].len() {
                                value += self.weights[m][n][k][l]
                                    * input[i + m][j + n][k];
                            }
                        }
                        values.push(value);
                    }
                    channel.push(values);
                }
                row.push(channel);
            }
            output.push(row);
        }
        output
    }
}

impl<T: Copy + Default> StyleLayer<T> {
    pub fn new(
        latent_dim: usize,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        upsample: bool,
        attention: bool,
    ) -> Result<Self, Box<dyn Error>> {
        let noise_injection = NoiseInjection::new()?; // Handle potential errors
        let adain = AdaIN::new(latent_dim, out_channels)?; // Handle potential errors
        let conv_weights = xavier_uniform(kernel_size, kernel_size, in_channels, out_channels);
        let conv = Conv2D::new(conv_weights, (1, 1), (kernel_size / 2, kernel_size / 2));

        let blur = if upsample {
            Some(Blur::new(out_channels)?) // Handle potential errors
        } else {
            None
        };

        let attention = if attention {
            Some(SelfAttention::new(out_channels)?) // Handle potential errors
        } else {
            None
        };

        Ok(StyleLayer {
            noise_injection,
            adain,
            conv,
            upsample,
            blur,
            attention,
        })
    }

    pub fn forward(&self, x: &[Vec<Vec<T>>], w: &[Vec<T>]) -> Result<Vec<Vec<Vec<Vec<T>>>>, Box<dyn Error>> {
        let mut x = if self.upsample {
            let x = upsample_2d(x)?; // Handle potential errors
            if let Some(blur) = &self.blur {
                blur.forward(&x)? // Handle potential errors
            } else {
                x
            }
        } else {
            x.to_vec()
        };

        x = self.conv.forward(&x);
        x = self.noise_injection.forward(x)?; // Handle potential errors
        x = self.adain.forward(x, w)?; // Handle potential errors
        leaky_relu(&mut x, 0.2);

        if let Some(attention) = &self.attention {
            attention.forward(&x)? // Handle potential errors
        } else {
            Ok(x)
        }
    }

    pub fn conv(&self) -> &Conv2D<T> {
        &self.conv
    }
}

fn xavier_uniform<T: Copy + Default>(
    kernel_size: usize,
    kernel_size: usize,
    in_channels: usize,
    out_channels: usize,
) -> Vec<Vec<Vec<Vec<T>>>> {
    let mut weights = Vec::new();
    let scale = (6.0 / ((kernel_size * kernel_size * in_channels + out_channels) as f64)).sqrt();
    for _ in 0..out_channels {
        let mut channel = Vec::new();
        for _ in 0..in_channels {
            let mut filter = Vec::new();
            for _ in 0..kernel_size {
                let mut row = Vec::new();
                for _ in 0..kernel_size {
                    row.push(rand::thread_rng().gen_range(-scale..scale));
                }
                filter.push(row);
            }
            channel.push(filter);
        }
        weights.push(channel);
    }
    weights
}

fn upsample_2d<T: Copy + Default>(input: &[Vec<Vec<T>>]) -> Result<Vec<Vec<Vec<T>>>, Box<dyn Error>> {
    let mut output = Vec::new();
    for row in input {
        let mut new_row = Vec::new();
        for i in 0..row.len() * 2 {
            let mut new_column = Vec::new();
            for j in 0..row[0].len() {
                let value = if i % 2 == 0 {
                    row[i / 2][j]
                } else {
                    T::default()
                };
                new_column.push(value);
            }
            new_row.push(new_column);
        }
        output.push(new_row);
    }

    for i in 0..input[0].len() * 2 {
        if i % 2 == 1 {
            let mut new_row = Vec::new();
            for _ in 0..output[0].len() {
                new_row.push(vec![T::default(); output[0][0].len()]);
            }
            output.insert(i, new_row);
        }
    }

    Ok(output)
}

fn leaky_relu<T: Copy + Default>(input: &mut Vec<Vec<Vec<Vec<T>>>>, negative_slope: f64) {
    for i in 0..input.len() {
        for j in 0..input[i].len() {
            for k in 0..input[i][j].len() {
                for l in 0..input[i][j][k].len() {
                    if input[i][j][k][l] < T::default() {
                        input[i][j][k][l] =
                            (input[i][j][k][l] as f64 * negative_slope) as T;
                    }
                }
            }
        }
    }
}