# LieRE: Generalizing Rotary Position Encodings

While Rotary Position Embeddings (RoPE) for natural language performs well and has become widely adopted, its adoption for other modalities has been slower. Here, we introduce Lie group Relative position Encodings (LieRE) that goes beyond RoPE in supporting higher dimensional inputs. We evaluate the performance of LieRE on 2D and 3D image classification tasks and observe that LieRE leads to marked improvements in performance (up to 6%), training efficiency (3.5x reduction), data efficiency (30%) compared to the baselines of DeiT III, RoPE-Mixed and Vision-Llama.


# Implementation for computing the rotation matrixes
We here share the code for implementing the rotation matrices. In short, every rotation matrix can be represented as the matrix exponential of a skew-symmetric matrix and we make the matrix learnable by parametrizing the rotations with generators before the matrix exponential.

# Base repo
We used the transformer implementation and default hyperparameters of https://github.com/kentaroy47/vision-transformers-cifar10.



