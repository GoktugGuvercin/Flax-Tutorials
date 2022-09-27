# Introduction to Autoencoders

Autoencoders are special type of neural network in deep learning; they are commonly used for regeneration of the input as output. An autoencoder is 
composed of two consecutive sub networks: Encoder and Decoder. Encoders try to condense the pieces of information in input into latent space in order to 
extract fewer but high level of features. These features are powerful global representatives of input context; hence, they are used by subsequent decoder
module to reconstruct the input as autoencoder output. Decoders are expected to behave in opposite manner compared to encoders, since their task is to 
reverse what encoders exactly do. Hence, decoder layers tend to extend and upsample compressed information in latent space into original input size.

<p align="center">
  <img src="https://github.com/GoktugGuvercin/Flax-Tutorials/blob/main/Image%20Denoising%20with%20Autoencoders/images/Autoencoder.png" />
</p>

Autoencoders go through information loss during the compression of input into latent space; hence, we cannot say that these architectures completely 
satisfy the condition  𝑔(𝑓(𝑥))=𝑥 in which the functions 𝑔 and 𝑓 represent decoder and encoder parts respectively. If the dimension of latent space is quite 
small compared with input, the loss of information becomes extreme and thereby preventing the decoder from reconstructing the input properly. This problem 
is remarkably observable in semantic segmentation; that's why, U-Net came up with skip connections that transfer low level details from encoder to decoder 
module; they partly avoid information loss. The deployment of high dimensional latent space is another solution to this problem; nonetheless, it may trigger overfitting because encoder cannot learn high level feature patterns sufficiently with large latent space.

The idea of autoencoder may sound meaningless at first sight because the attempt to learn  𝑔(𝑓(𝑥))=𝑥 seems useless. However, encoder part of autoencoders 
trained for input reconstruction converts into powerful feature extractor for dimensionality reduction and feature learning. Activation functions used in 
encoder and decoder modules like ReLU qualify the entire model to have non-linearity, which makes the encoder to be adjusted to non-linear generalization 
of Principal Component Analysis (PCA). In addition to this, autoencoders can be used for generative modelling, which is called Variational Autoencoders. 
To get into detail, KL divergence applied on reconstruction loss of autoencoders (L2 and L1 loss in general) can approximate feature space encoded in 
latent representation to gaussian probability distribution. What we actually do at this point is to put a constraint at latent space of autoencoder in 
order to get the latent into particular form. As a result, the obtained distribution in latent space is used to derive new samples with distinct context.
