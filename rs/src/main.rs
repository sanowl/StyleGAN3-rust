use tch::{nn, Device, Kind, Tensor};

mod adain;
mod blur;
mod discriminator;
mod generator;
mod mapping_network;
mod noise_injection;
mod residual_block;
mod self_attention;
mod style_layer;

use discriminator::Discriminator;
use generator::Generator;

fn main() {
    let device = Device::cuda_if_available();
    println!("Using device: {:?}", device);

    let latent_dim = 512;
    let hidden_dim = 512;
    let output_channels = 3;
    let num_layers = 8;
    let batch_size = 32;
    let num_epochs = 100;
    let learning_rate = 0.0002;
    let num_batches = 100; // Define the number of batches per epoch

    let generator = Generator::new(latent_dim, hidden_dim, output_channels, num_layers).to(device);
    let discriminator = Discriminator::new(output_channels, hidden_dim, num_layers).to(device);

    let mut generator_optimizer = nn::Adam::default()
        .build(&generator.parameters(), learning_rate)
        .unwrap();
    let mut discriminator_optimizer = nn::Adam::default()
        .build(&discriminator.parameters(), learning_rate)
        .unwrap();

    for epoch in 0..num_epochs {
        for _ in 0..num_batches {
            // Generate random latent vectors
            let z = Tensor::randn(&[batch_size, latent_dim], (Kind::Float, device));

            // Generate fake images
            let fake_images = generator.forward(&z);

            // Assume `real_images` is a batch of real images from your dataset
            let real_images = Tensor::randn(
                &[batch_size, output_channels, 64, 64],
                (Kind::Float, device),
            ); // Replace with actual data

            // Discriminate real and fake images
            let real_scores = discriminator.forward(&real_images);
            let fake_scores = discriminator.forward(&fake_images);

            // Compute losses
            let generator_loss = -fake_scores.mean(Kind::Float); // Simplified loss for illustration
            let discriminator_loss =
                (fake_scores - 1).pow(2).mean(Kind::Float) + real_scores.pow(2).mean(Kind::Float); // Simplified loss for illustration

            // Update generator
            generator_optimizer.zero_grad();
            generator_loss.backward();
            generator_optimizer.step();

            // Update discriminator
            discriminator_optimizer.zero_grad();
            discriminator_loss.backward();
            discriminator_optimizer.step();
        }

        // Print losses and generate sample images
        println!(
            "Epoch [{}/{}], Generator Loss: {:.4}, Discriminator Loss: {:.4}",
            epoch + 1,
            num_epochs,
            f64::from(&generator_loss),
            f64::from(&discriminator_loss)
        );

        // Generate and save sample images
        let sample_z = Tensor::randn(&[1, latent_dim], (Kind::Float, device));
        let sample_images = generator.forward(&sample_z);
        sample_images
            .save(format!("sample_epoch_{}.png", epoch))
            .unwrap(); // Assuming sample_images is in a format that can be saved directly
    }
}
