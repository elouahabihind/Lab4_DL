### Lab 4: Autoencoder, Variational Autoencoder, and GANs

---

#### Objective:
The objective of this report is to compare and analyze the performance of Autoencoder (AE), Variational Autoencoder (VAE), and Generative Adversarial Networks (GANs) in generating new images.

---

### Part 1: Autoencoder (AE) and Variational Autoencoder (VAE)

#### Definition of Models:

1. **Autoencoder (AE):**
   - **Encoder Architecture:** Consists of several fully connected layers with ReLU activation functions.
   - **Decoder Architecture:** Also composed of fully connected layers with ReLU activation except for the output layer which uses a Sigmoid activation to normalize values between 0 and 1.
   - **Training:** Trained using Mean Squared Error (MSE) loss function.
   
2. **Variational Autoencoder (VAE):**
   - **Encoder Architecture:** Similar to AE but includes two additional fully connected layers for computing mean (μ) and log variance (logvar) of the latent space.
   - **Reparameterization:** Uses a reparameterization trick to sample from the learned distribution in the latent space.
   - **Decoder Architecture:** Similar to AE.
   - **Training:** Combines Reconstruction Loss (BCE) and Kullback-Leibler Divergence (KLD) loss.



- **Autoencoder (AE):**
  - Achieved a final loss of 0.0320 after 15 epochs of training.

- **Variational Autoencoder (VAE):**
  - Achieved a final loss of 139.9514 with a KL Divergence of 6.7920 after 15 epochs of training.

---

### Part 2: Generative Adversarial Networks (GANs)

#### Definition of Models:

1. **Discriminator Architecture:**
   - Sequential model with convolutional layers followed by batch normalization and LeakyReLU activation.
   - Final layer outputs a single value representing the probability of the input being real or fake.

2. **Generator Architecture:**
   - Sequential model with transpose convolutional layers followed by batch normalization and ReLU activation.
   - Final layer outputs generated images with Tanh activation.


#### Results of Training:

- **Losses:**
  - **Discriminator Loss (Loss_disc):** Decreased gradually over epochs, indicating improved discrimination between real and fake images.
  - **Generator Loss (Loss_gene):** Fluctuated but showed a decreasing trend overall, signifying the generator's learning to produce more convincing images.

---

### Conclusion:

- Autoencoder and Variational Autoencoder are effective in reconstructing images with AE achieving a lower loss compared to VAE.
- VAE introduces a trade-off between reconstruction loss and KL Divergence, resulting in a higher overall loss.
- GANs demonstrate the capability to generate new images, with the discriminator and generator losses showing a dynamic interplay during training.
- Each model has its strengths and weaknesses, making them suitable for different image generation tasks based on specific requirements and constraints.
