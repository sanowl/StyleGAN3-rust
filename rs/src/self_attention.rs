use std::ops::{Deref, DerefMut};

struct Conv2D<T> {
    weights: Vec<Vec<Vec<Vec<T>>>>,
    bias: Vec<T>,
    stride: (usize, usize),
    padding: (usize, usize),
}

impl<T: Copy + Default + std::ops::Mul<Output = T> + std::ops::Add<Output = T>> Conv2D<T> {
    fn new(
        weights: Vec<Vec<Vec<Vec<T>>>>,
        bias: Vec<T>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Self {
        Conv2D {
            weights,
            bias,
            stride,
            padding,
        }
    }

    fn forward(&self, input: &[Vec<Vec<T>>]) -> Vec<Vec<Vec<T>>> {
        let (input_channels, input_height, input_width) = (input.len(), input[0].len(), input[0][0].len());
        let (output_channels, kernel_height, kernel_width) = (
            self.weights.len(),
            self.weights[0][0].len(),
            self.weights[0][0][0].len(),
        );

        let mut output = vec![vec![vec![T::default(); input_width - kernel_width + 1 + 2 * self.padding.1];
                                   input_height - kernel_height + 1 + 2 * self.padding.0];
                              output_channels];

        for out_c in 0..output_channels {
            for out_h in 0..output[0].len() {
                for out_w in 0..output[0][0].len() {
                    let mut sum = T::default();
                    for in_c in 0..input_channels {
                        for kh in 0..kernel_height {
                            for kw in 0..kernel_width {
                                let input_h = out_h - self.padding.0 as isize + kh as isize;
                                let input_w = out_w - self.padding.1 as isize + kw as isize;
                                if input_h >= 0
                                    && input_h < input_height as isize
                                    && input_w >= 0
                                    && input_w < input_width as isize
                                {
                                    let input_val = input[in_c][input_h as usize][input_w as usize];
                                    let weight_val = self.weights[out_c][in_c][kh][kw];
                                    sum = sum + input_val * weight_val;
                                }
                            }
                        }
                    }
                    output[out_c][out_h][out_w] = sum + self.bias[out_c];
                }
            }
        }

        output
    }
}

pub struct SelfAttention<T> {
    query: Conv2D<T>,
    key: Conv2D<T>,
    value: Conv2D<T>,
    gamma: Vec<T>,
}

impl<T: Copy + Default + std::ops::Mul<Output = T> + std::ops::Add<Output = T>> SelfAttention<T> {
    pub fn new(in_channels: usize) -> Self {
        let query_weights = vec![
            vec![vec![vec![T::default(); in_channels / 8]; in_channels / 8]; 1];
            1
        ];
        let query_bias = vec![T::default(); in_channels / 8];
        let query = Conv2D::new(query_weights, query_bias, (1, 1), (0, 0));

        let key_weights = vec![
            vec![vec![vec![T::default(); in_channels / 8]; in_channels]; 1];
            1
        ];
        let key_bias = vec![T::default(); in_channels / 8];
        let key = Conv2D::new(key_weights, key_bias, (1, 1), (0, 0));

        let value_weights = vec![
            vec![vec![vec![T::default(); in_channels]; in_channels]; 1];
            1
        ];
        let value_bias = vec![T::default(); in_channels];
        let value = Conv2D::new(value_weights, value_bias, (1, 1), (0, 0));

        let gamma = vec![T::default(); 1];

        SelfAttention {
            query,
            key,
            value,
            gamma,
        }
    }

    pub fn forward(&self, x: &[Vec<Vec<T>>]) -> Vec<Vec<Vec<T>>> {
        let (batch_size, channels, height, width) = (x.len(), x[0].len(), x[0][0].len(), x[0][0][0].len());

        let query = self.query.forward(x);
        let query = view_and_permute(&query, batch_size, channels / 8, width * height);

        let key = self.key.forward(x);
        let key = view(&key, batch_size, channels / 8, width * height);

        let attention = batched_matrix_multiplication(&query, &key.transpose());
        let attention = softmax(&attention, -1);

        let value = self.value.forward(x);
        let value = view(&value, batch_size, channels, width * height);

        let out = batched_matrix_multiplication(&value, &attention.transpose());
        let out = view_and_reshape(&out, batch_size, channels, height, width);

        let gamma_tensor = repeat(&self.gamma, batch_size, channels, height, width);
        element_wise_addition(&element_wise_multiplication(&gamma_tensor, &out), x)
    }
}

// Helper functions

fn view_and_permute<T: Copy>(tensor: &[Vec<Vec<T>>], batch_size: usize, channels: usize, height_width: usize) -> Vec<Vec<Vec<T>>> {
    let mut reshaped = Vec::with_capacity(batch_size);
    for batch in tensor.chunks(channels) {
        let mut batch_tensor = Vec::with_capacity(height_width);
        for hw in batch.chunks(height_width) {
            batch_tensor.push(hw.to_vec());
        }
        reshaped.push(batch_tensor);
    }
    let mut permuted = Vec::with_capacity(batch_size);
    for batch in reshaped {
        permuted.push(batch.into_iter().flatten().collect());
    }
    permuted
}

fn view<T: Copy>(tensor: &[Vec<Vec<T>>], batch_size: usize, channels: usize, height_width: usize) -> Vec<Vec<Vec<T>>> {
    let mut reshaped = Vec::with_capacity(batch_size);
    for batch in tensor.chunks(channels) {
        let mut batch_tensor = Vec::with_capacity(height_width);
        for hw in batch.chunks(height_width) {
            batch_tensor.push(hw.to_vec());
        }
        reshaped.push(batch_tensor);
    }
    reshaped
}

fn batched_matrix_multiplication<T: Copy + std::ops::Mul<Output = T> + std::ops::Add<Output = T>>(
    a: &[Vec<T>],
    b: &[Vec<T>],
) -> Vec<Vec<T>> {
    let (a_rows, a_cols) = (a.len(), a[0].len());
    let (b_rows, b_cols) = (b.len(), b[0].len());

    assert_eq!(a_cols, b_rows, "Matrices are not compatible for multiplication");

    let mut result = vec![vec![T::default(); b_cols]; a_rows];

    for i in 0..a_rows {
        for j in 0..b_cols {
            for k in 0..a_cols {
                result[i][j] = result[i][j] + a[i][k] * b[k][j];
            }
        }
    }

    result
}

fn softmax<T: Copy + std::ops::Mul<Output = T> + std::ops::Add<Output = T>>(tensor: &[Vec<T>], dim: isize) -> Vec<Vec<T>> {
    let (rows, cols) = (tensor.len(), tensor[0].len());
    let mut result = tensor.to_vec();

    if dim == 0 {
        for j in 0..cols {
            let mut max_val = result[0][j];
            for i in 1..rows {
                max_val = max_val.max(result[i][j]);
            }
            let mut row_sum = T::default();
            for i in 0..rows {
                result[i][j] = (result[i][j] - max_val).exp();
                row_sum = row_sum + result[i][j];
            }
            for i in 0..rows {
                result[i][j] = result[i][j] / row_sum;
            }
        }
    } else if dim == 1 {
        for i in 0..rows {
            let mut max_val = result[i][0];
            for j in 1..cols {
                max_val = max_val.max(result[i][j]);
            }
            let mut col_sum = T::default();
            for j in 0..cols {
                result[i][j] = (result[i][j] - max_val).exp();
                col_sum = col_sum + result[i][j];
            }
            for j in 0..cols {
                result[i][j] = result[i][j] / col_sum;
            }
        }
    }

    result
}

fn view_and_reshape<T: Copy>(tensor: &[Vec<T>], batch_size: usize, channels: usize, height: usize, width: usize) -> Vec<Vec<Vec<T>>> {
    let mut reshaped = Vec::with_capacity(batch_size);
    for batch in tensor.chunks(channels * height * width) {
        let mut batch_tensor = Vec::with_capacity(channels);
        for channel in batch.chunks(height * width) {
            let mut channel_tensor = Vec::with_capacity(height);
            for h in channel.chunks(width) {
                channel_tensor.push(h.to_vec());
            }
            batch_tensor.push(channel_tensor);
        }
        reshaped.push(batch_tensor);
    }
    reshaped
}

fn repeat<T: Copy>(tensor: &[T], batch_size: usize, channels: usize, height: usize, width: usize) -> Vec<Vec<Vec<Vec<T>>>> {
    let mut repeated = Vec::with_capacity(batch_size);
    for _ in 0..batch_size {
        let mut batch_tensor = Vec::with_capacity(channels);
        for _ in 0..channels {
            let mut channel_tensor = Vec::with_capacity(height);
            for _ in 0..height {
                channel_tensor.push(tensor.to_vec());
            }
            batch_tensor.push(channel_tensor);
        }
        repeated.push(batch_tensor);
    }
    repeated
}

fn element_wise_multiplication<T: Copy + std::ops::Mul<Output = T>>(a: &[Vec<Vec<Vec<T>>>], b: &[Vec<Vec<T>>]) -> Vec<Vec<Vec<T>>> {
    let mut result = Vec::with_capacity(a.len());
    for (batch_a, batch_b) in a.iter().zip(b.iter()) {
        let mut batch_result = Vec::with_capacity(batch_a.len());
        for (channel_a, channel_b) in batch_a.iter().zip(batch_b.iter()) {
            let mut channel_result = Vec::with_capacity(channel_a.len());
            for (row_a, row_b) in channel_a.iter().zip(channel_b.iter()) {
                channel_result.push(row_a.iter().zip(row_b.iter()).map(|(a, b)| a * b).collect());
            }
            batch_result.push(channel_result);
        }
        result.push(batch_result);
    }
    result
}

fn element_wise_addition<T: Copy + std::ops::Add<Output = T>>(a: &[Vec<Vec<Vec<T>>>], b: &[Vec<Vec<T>>]) -> Vec<Vec<Vec<T>>> {
    let mut result = Vec::with_capacity(a.len());
    for (batch_a, batch_b) in a.iter().zip(b.iter()) {
        let mut batch_result = Vec::with_capacity(batch_a.len());
        for (channel_a, channel_b) in batch_a.iter().zip(batch_b.iter()) {
            let mut channel_result = Vec::with_capacity(channel_a.len());
            for (row_a, row_b) in channel_a.iter().zip(channel_b.iter()) {
                channel_result.push(row_a.iter().zip(row_b.iter()).map(|(a, b)| a + b).collect());
            }
            batch_result.push(channel_result);
        }
        result.push(batch_result);
    }
    result
}