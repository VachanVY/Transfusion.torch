# Transfusion
## Introduction
* ![image](https://github.com/user-attachments/assets/342f8647-e4bc-48bc-99ce-d53332e725b8)
* Transfusion by pretraining a transformer model on 50% text and 50% image data using a different objective for each modality: next token prediction for text and diffusion for images
* We apply causal attention for text tokens and bidirectional
attention for image patches. For inference, we introduce a decoding algorithm that combines the
standard practices of text generation from language models and image generation from diffusion
models
* Intra-image bidirectional attention is important, and replacing it with causal
attention hurts text-to-image generation

## Language Modelling Utils and Loss
* Autoregressive Classification
* Usual Cross-Entropy Loss

## Diffusion Utils and Loss
* ![image](https://github.com/user-attachments/assets/7d03be4a-4426-4191-8ebb-5cb95fc5faac)
* Noise Schedule: cosine scheduler
  (We found that while the linear noise schedule used in Ho et al. (2020) worked well for high-resolution images, it was sub-optimal for images of 
   resolution 64 × 64 and 32 × 32)
  ![image](https://github.com/user-attachments/assets/7e9bbb6f-5cb9-4b23-aa7c-a42e7fbd03e1)
* Loss: Mean Squared Error
* Latent Image Representation: Variational autoencoders (VAEs) [Kingma and Welling, 2013] can save compute by
encoding images into a lower-dimensional latent space

## Data Representation
* Discrete text and continuous images
* Each text string is tokenized into a sequence of discrete tokens from a fixed vocabulary,
where each token is represented as an integer

## Model Architecture
* ![image](https://github.com/user-attachments/assets/a185a2ed-3459-4030-9b90-78ae30e75b1d)
* The vast majority of the model’s parameters belong to a single transformer,
which processes every sequence, regardless of modality (We follow Llama’s [Touvron et al., 2023a] flavour of the transformer block, which includes the SwiGLU
activation function [Shazeer, 2020] and RoPE [Su et al., 2024])
* To convert our data into this space, we use lightweight modality-specific components with unshared parameters
* For text, these are the embedding matrices
* Images, we experiment with two alternatives for compressing local windows of k × k patch vectors into a single transformer vector (and vice versa):
  1. a simple linear layer (We add an embedding of the timestep t to every patch vector before the linear layer)
  2. up and down blocks of a U-Net (We replace the U-Net’s AdaLayerNorm with regular layer norm in our implementation)
* Transfusion Attention: While text is naturally sequential, images are not, and are usually
modelled with unrestricted (bidirectional) attention. Transfusion combines both attention patterns
by applying causal attention to every element in the sequence, and bidirectional attention within the
aspects of each individual image

## Training Objective
* LM loss is computed per token (When the input is a BOI token, we do not compute any loss), while diffusion loss is computed per image, which may span multiple
elements (image patches) in the sequence
* Specifically, we add noise ϵ to each input latent image
x0 according to the diffusion process to produce xt before patchification, and then compute the
image-level diffusion loss
* ![image](https://github.com/user-attachments/assets/75015697-691d-452a-8b20-23b3d4fbe7e6)
* ![image](https://github.com/user-attachments/assets/289d4252-1ebc-4298-8086-a0fcc5b675a3)


## Optimization
* AdamW => | betas=(0.9, 0.95) | eps=1e-8 | lr=3e-4 | warmup=4000 | min_lr=1.5e-5 | weight_decay=0.1 | clip_norm=1.0 |
* balancing_coeff (lambda in loss function) = 5

## Inference
* ![image](https://github.com/user-attachments/assets/f6e7969e-02a6-416d-90e4-ae6dca3b3c93)
* 250 diffusion steps (but trained on 1000 timesteps)
* cfg_coeff = 5.0

---
![image](https://github.com/user-attachments/assets/53992abe-c322-4031-a49f-d48924c9e52d)

---
![image](https://github.com/user-attachments/assets/70afb4df-cf92-47b3-b3a2-c74a6f6310a6)
---
