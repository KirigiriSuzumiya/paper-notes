# MolScribe: Robust Molecular Structure Recognition with Image-to-Graph Generation

paper:https://pubs.acs.org/doi/10.1021/acs.jcim.2c01480

code: https://github.com/thomas0809/MolScribe

## Task Formulation

Molecular structure recognition is the task of translating single-molecule images into corresponding molecular structures. In this paper,we formulate it as image-to-graph generation. 

|                         |                         |                        |
| ----------------------- | ----------------------- | ---------------------- |
| I                       |                        | single-molecule image |
| G                | G= (A,B)        | 2D molecular graph     |
| A | A= {$a_1,a_2, ...,an$} | set of atoms |
| $a_i$ | $a_i= (l_i,x_i,y_i)$ | atom |
| $l_i$ |  | the atom’s corresponding SMILES (sub)string |
| $x_i,y_i$ |  | 2D coordinates of the atom in the image |
| B | B⊂A×A×T | set of bonds |
| T | T= {single,double,triple,aromatic,None} | set of bond types |

## Model

![](C:\Users\buy1\Desktop\images_large_ci2c01480_0003.jpeg)

The input image is encoded with an image encoder,and the graph decoder predicts the atoms and bonds. A molecular graph is  constructed from the predictions and converted to a MOLfile or a SMILES string.

**image-to-graph translation**
$$
P(G|I)=P(A|I)P(B|A,I)
$$

### Image Encoder

The image encoder is a Swin Transformer,a state-of-the-art model in many computer vision tasks. We use the Swin-B model, which has 88 M parameters in total and pretrained on ImageNet-22K.

### Atom predictor 

an **autoregressive decoder**  **$P(A|I)$**  that generates the atom in a sequence **$S^A$**

The decoder is a 6-layer Transformer with 8 attention heads, a hidden dimension of 256,and sinusoidal positional encoding.
$$
P(A|I)=P(S^A|I)=\prod_{i=1}^{n}P(S^A_i|S^A_{<i},I) \\
S^A=[l_1,\hat{x}_1,\hat{y}_1,l_2,\hat{x}_2,\hat{y}_2,\cdots,l_n,\hat{x}_n,\hat{y}_n] \\
\hat{x}_i=[\frac{x_i}{W} \times n_{bins},],\hat{y}_i=[\frac{y_i}{H} \times n_{bins},]
$$

### Bond predictor 

The bond predictor is a **feed forward network** that predicts the bond between each pair of atoms.  Each atom $a_i$ is represented as a vector $h_{a_i}$, the hidden state of its last token in the decoder output.

The bond predictor is a 2-layer feedforward network with ReLU activation on top of the decoder and has the same hidden dimension.
$$
P(B|A,I)=\prod_{i=1}^{n}\prod_{j=1}^{n}P(b_{i,j}|A,I)
$$
For **symmetric bonds**, $b_{i,j}$ and $b_{j,i}$ are expected to be the same, so the probabilities are averaged
$$
\hat{P}(b_{i,j}=t)=\frac{1}{2}(P(b_{i,j}=t)+P(b_{i,j}=t))
$$
For **asymmetric bonds** (wedges), as a solid wedge(`s.w.`)is equivalent to a dashed wedge(`d.w.`)in the opposite direction, we set$b_{i,j}$= "s.w." and $b_{j,i}$ = "d.w." if there is a solid wedge from $a_i$ to $a_j$ and vice versa.
$$
\hat{P}(b_{i,j}= s.w.)=\frac{1}{2}(P(b_{i,j}=s.w.)+P(b_{j,i}=d.w.)) \\
\hat{P}(b_{i,j}= d.w.)=\frac{1}{2}(P(b_{i,j}=d.w.)+P(b_{j,i}=s.w.))
$$
