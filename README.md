# Transfusion [[Paper]](https://arxiv.org/pdf/2408.11039v1)
* Transfusion is a Multi-Modal Transformer, it can generate text like GPTs and images like Diffusion Models, all at once in one go not separately!
* It can easily switch between text and image modalities for generations, and it is nothing complicated, just a single transformer with some modality-specific components!
* This can easily be extended to other modalities like videos, audio, etc, but for now, it can only take images and text as input
<!-- * For now I have **"test-trained"** it on
  * Fashion MNIST Dataset (contains images of Fashion Items like T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
  * MNIST Dataset (contains images of Digits from 0 to 9)
  * I have taken the classes as text and trained it. See below for some generated examples... -->
* **`TODO`: Train on a large Multi-Modal Dataset (something like tiny stories dataset with images in between illustrating the story...?)**

```python
from src.llama2c import Transformer as LLaMA

class config:
    ... # Fill in some parameters for the model | see src/configs.py for reference

model = Transfussion(
    model=LLaMA(config),
    config=config
)

text_and_images = [
    [
        torch.randint(0, 10, (39,)), # text
        # You get "image" after passing the image to PatchOps.patchify() while preprocessing
        (torch.randn(345, config.patch_size**2 * config.in_channels), torch.randint(0, config.num_timesteps, (1,))), # (image, timestep)
        torch.randint(0, 10, (14,)) # text
    ],
    [
        torch.randint(0, 10, (16,)), # text
        # You get "image" after passing the image to PatchOps.patchify() while preprocessing
        (torch.randn(359, config.patch_size**2 * config.in_channels), torch.randint(0, config.num_timesteps, (1,))), # (image, timestep)
        torch.randint(0, 10, (5,)), # text
        # You get "image" after passing the image to PatchOps.patchify() while preprocessing
        (torch.randn(2, config.patch_size**2 * config.in_channels), torch.randint(0, config.num_timesteps, (1,))),   # (image, timestep)
        torch.randint(0, 10, (9,))  # text
    ]
]
output = model(text_and_images, [["text", "image", "text"], ["text", "image", "text", "image", "text"]])
```

## Contents
<!-- * [Test Trained on Fashion MNIST Dataset](https://github.com/VachanVY/Transfusion.torch/tree/main?tab=readme-ov-file#test-trained-on-fashion-mnist-dataset) <===> [Training Notebook with some generated samples](https://github.com/VachanVY/Transfusion.torch/blob/main/fashion_mnist_test_transfusion.ipynb)
* [Test Trained on MNIST dataset](https://github.com/VachanVY/Transfusion.torch/tree/main?tab=readme-ov-file#test-trained-on-mnist-dataset) <===> [Training Notebook with some generated samples](https://github.com/VachanVY/Transfusion.torch/blob/main/mnist_test_transfusion.ipynb) -->
* [Important Snippets from the Paper](https://github.com/VachanVY/Transfusion.torch/tree/main?tab=readme-ov-file#introduction)
  
<!-- ## Test Trained on Fashion MNIST Dataset
* Can produce 2 images of Fashion Items along with the text (in the form of tokens) shown above the respective images
  <!-- the integers above the images can be interpreted using this dictionary -->
  <!-- ```python
  {'T-shirt/top': 0,
    'Trouser': 1,
    'Pullover': 2,
    'Dress': 3,
    'Coat': 4,
    'Sandal': 5,
    'Shirt': 6,
    'Sneaker': 7,
    'Bag': 8,
    'Ankle boot': 9}
  ``` -->
  <!-- So `5` means it's a sandal and `0` means it's a T-shirt/top from the below image and just like that some more examples. Use the dictionary to interpret the tokens as text (for now, will change it)\
  ![download](https://github.com/user-attachments/assets/113abcdd-6de5-4c9e-81e6-d9a7b2671293)
  
  ---
  `8` is a bag\
  ![download](https://github.com/user-attachments/assets/6dca45c8-68be-45d8-83fa-9fa90e0a5b11)
* See [this notebook](https://github.com/VachanVY/Transfusion.torch/blob/main/fashion_mnist_test_transfusion.ipynb) for more examples.

## Test Trained on MNIST dataset
* Generates text and images in an alternating way as shown below
    
  ![download](https://github.com/user-attachments/assets/bcd1c1dd-2225-4de1-ad13-e01da6c0fc5c)
  ---
  ![download](https://github.com/user-attachments/assets/df825844-7629-44b4-9b32-06c9d66a7198)
* See [this notebook](https://github.com/VachanVY/Transfusion.torch/blob/main/mnist_test_transfusion.ipynb) for more examples -->

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
