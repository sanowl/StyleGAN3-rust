use rand::distributions::{Distribution, Normal};

pub struct NoiseInjection<T> {
    scale: T,
}

impl<T: Copy + Default + std::ops::Mul<Output = T>> NoiseInjection<T> {
    pub fn new() -> Self {
        NoiseInjection {
            scale: T::default(),
        }
    }

    pub fn forward(&self, x: &[Vec<Vec<T>>]) -> Vec<Vec<Vec<T>>> {
        let mut output = x.to_vec();
        let normal = Normal::new(T::default(), self.scale).unwrap();

        for c in 0..output.len() {
            for h in 0..output[c].len() {
                for w in 0..output[c][h].len() {
                    let noise = normal.sample(&mut rand::thread_rng());
                    output[c][h][w] = output[c][h][w] + noise;
                }
            }
        }

        output
    }
}
