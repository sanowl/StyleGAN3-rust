use std::error::Error;
use std::fmt;

#[derive(Debug)]
pub enum BlurError {
    InvalidChannels(String),
    InvalidInput(String),
}

impl fmt::Display for BlurError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BlurError::InvalidChannels(msg) => write!(f, "Invalid channels: {}", msg),
            BlurError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl Error for BlurError {}

pub struct Blur<T> {
    kernel: Vec<Vec<Vec<Vec<T>>>>,
}

impl<T: Copy + Default + std::ops::Mul<Output = T> + std::ops::Div<Output = T>> Blur<T> {
    pub fn new(channels: usize) -> Result<Self, BlurError> {
        if channels == 0 {
            return Err(BlurError::InvalidChannels(
                "Channels must be a positive integer".to_string(),
            ));
        }

        let kernel_1d = vec![
            T::default(),
            T::from(2.0).unwrap(),
            T::from(4.0).unwrap(),
            T::from(2.0).unwrap(),
            T::default(),
        ];

        let mut kernel = Vec::with_capacity(channels);
        for _ in 0..channels {
            let mut channel = Vec::with_capacity(1);
            let mut filter = Vec::with_capacity(3);
            for _ in 0..3 {
                let mut row = Vec::with_capacity(3);
                for weight in &kernel_1d {
                    row.push(*weight);
                }
                filter.push(row);
            }
            channel.push(filter);
            kernel.push(channel);
        }

        for channel in &mut kernel {
            for filter in channel {
                for row in filter {
                    for weight in row {
                        *weight /= T::from(16.0).unwrap();
                    }
                }
            }
        }

        Ok(Blur { kernel })
    }

    pub fn forward(&self, input: &[Vec<Vec<T>>]) -> Result<Vec<Vec<Vec<T>>>, BlurError> {
        let (input_channels, input_height, input_width) =
            (input.len(), input[0].len(), input[0][0].len());

        if input_channels != self.kernel.len() {
            return Err(BlurError::InvalidInput(
                "Input tensor must have the same number of channels as the kernel".to_string(),
            ));
        }

        let mut output = vec![vec![vec![T::default(); input_width]; input_height]; input_channels];

        for c in 0..input_channels {
            for h in 0..input_height {
                for w in 0..input_width {
                    let mut sum = T::default();
                    for kh in 0..3 {
                        for kw in 0..3 {
                            let input_h = h as isize - 1 + kh as isize;
                            let input_w = w as isize - 1 + kw as isize;
                            if input_h >= 0
                                && input_h < input_height as isize
                                && input_w >= 0
                                && input_w < input_width as isize
                            {
                                sum = sum
                                    + input[c][input_h as usize][input_w as usize]
                                        * self.kernel[c][0][kh][kw];
                            }
                        }
                    }
                    output[c][h][w] = sum;
                }
            }
        }

        Ok(output)
    }
}
