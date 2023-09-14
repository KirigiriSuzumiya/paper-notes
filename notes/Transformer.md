# Attention Is All You Need

paper:https://arxiv.org/abs/1706.03762

##  Model Architecture

 the encoder maps an input sequence of symbol representations$ (x_1, ..., x_n)$ to a sequence of continuous representations $z = (z_1, ..., z_n)$. Given z, the decoder then generates an output sequence $(y_1, ..., y_m)$ of symbols one element at a time. At each step the model is auto-regressive , consuming the previously generated symbols as additional input when generating the next.

![image-20230914092405281](C:\Users\buy1\Desktop\assets\image-20230914092405281.png)

### Encoder and Decoder Stacks

#### Encoder

- composed of a stack of $N = 6$ identical layers
-  Each layer has two sub-layers
  - a multi-head self-attention mechanism
  - position-wise fully connected feed-forward network
- a residual connection around each of the two sub-layers, followed by layer normalization
- all sub-layers in the model produce outputs of dimension $d_{model} = 512$.

#### Decoder

- composed of a stack of $N = 6$ identical layers
- inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack
- modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions
  - ensures that the predictions for position $i$ can depend only on the known outputs at positions less than $i$.

### Attention

![image-20230914093717853](C:\Users\buy1\Desktop\assets\image-20230914093717853.png)

#### Scaled Dot-Product Attention

The input consists of queries and keys of dimension $d_k$, and values of dimension $d_v$. We compute the dot products of the query with all keys, divide each by $\sqrt{d_k}$, and apply a softmax function to obtain the weights on the values.
$$
Attention(Q,K,V)= softmax(\frac{QK^T}{\sqrt{d_k}})V
$$

#### Multi-Head Attention

Instead of performing a single attention function with $d_{model}$-dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values $h$ times with different, learned linear projections to $d_k, d_k \  and \ d_v$ dimensions, respectively
$$
MultiHead(Q,K,V)= Concat(head_1,\dots,head_h)W^O \\
$$
where $ head_i=Attention(QW^Q_i,KW^K_i,VW^V_i)$ 

and the projections are parameter matrices $ W^Q_i \in \real^{d_{model}\times d_k},W^K_i \in \real^{d_{model}\times d_k},W^V_i \in \real^{d_{model}\times d_v},W^O \in \real^{hd_v\times d_{model}}$

In this work we employ $h = 8$ parallel attention layers, or heads. For each of these we use $d_k = d_v = d_{model}/h = 64$. Due to the reduced dimension of each head, the total computational cost is similar to that of single-head attention with full dimensionality.

### Position-wise Feed-Forward Networks

This consists of two linear transformations with a ReLU activation in between.
$$
FFN(x)=max(0,xW_1+b_1)W_2+b_2
$$

### Positional Encoding

The positional encodings have the same dimension $d_{model}$ as the embeddings, so that the two can be summed.
$$
PE_{(pos,2i)}=sin(pos/10000^{2i/d_{model}}) \\
PE_{(pos,2i+1)}=cos(pos/10000^{2i/d_{model}})
$$
where $pos$ is the position and $i$ is the dimension. That is, each dimension of the positional encoding corresponds to a sinusoid. 

## Training

### Optimizer

We used the Adam optimizer with $β1 = 0.9, β2 = 0.98 \ and \  ϵ = 10^{−9}$ . We varied the learning rate over the course of training, according to the formula:
$$
lrate=d^{-0.5}_{model}\times \min(step\_num^{-0.5},step\_num\times warmup\_steps^{-1.5})
$$
This corresponds to increasing the learning rate linearly for the first warmup_steps training steps, and decreasing it thereafter proportionally to the inverse square root of the step number. We used $warmup\_steps = 4000$.