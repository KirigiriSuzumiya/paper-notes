# Swin Transformer: Hierarchical Vision Transformer using Shifted Windows

![teaser](https://github.com/microsoft/Swin-Transformer/raw/main/figures/teaser.png)

| Vision                      | Language                  | Difficulty                                                   |
| --------------------------- | ------------------------- | ------------------------------------------------------------ |
| vary substantially in scale | word token                | fixed scale in NLP while vary in CV                          |
| resolution of pixels        | words in passages of text | computational complexity of its self-attention is quadratic to image size |

Swin Transformer, which capably serves as a general-purpose backbone for computer vision. It is basically a hierarchical Transformer whose representation is computed with shifted windows. The shifted windowing scheme brings greater efficiency by limiting self-attention computation to non-overlapping local windows while also allowing for cross-window connection.

##  Overall Architecture

- first splits an input RGB image into non-overlapping patches by a patch splitting module

  - use a patch size of 4 × 4 and thus the feature dimension of each patch is 4 × 4 × 3 = 48

- Stage 1

  - A linear embedding layer is applied on this raw-valued feature to project it to an arbitrary dimension (denoted as C)
  - Several Transformer blocks with modified self-attention computation (Swin Transformer blocks) are applied on these patch tokens

- Stage 2, 3 &4

  -  To produce a hierarchical representation, the number of tokens is reduced by patch merging layers 
  -  first patch merging layer concatenates the features of each group of 2 × 2 neighboring patches
    - applies a linear layer on the 4C-dimensional concatenated features. 
    - This reduces the number of tokens by a multiple of 2×2 = 4 (2× downsampling of resolution), and the output dimension is set to 2C.
  - Swin Transformer blocks are applied afterwards for feature transformation

  

### Swin Transformer block

Swin Transformer is built by replacing the standard multi-head self attention (MSA) module in a Transformer block by a module based on shifted windows, with other layers kept the same. A Swin Transformer block consists of a shifted window based MSA module, followed by a 2-layer MLP with GELU nonlinearity in between. A LayerNorm (LN) layer is applied before each MSA module and each MLP, and a residual connection is applied after each module.

### Shifted Window based Self-Attention
