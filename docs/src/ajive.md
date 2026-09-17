# Angle-Based Joint and Individual Variation Explained (AJIVE)

AJIVE decomposes multiple data blocks measured on the same set of samples into  **joint**, **individual**, and **residual** variation. The method focuses on the 
**score subspaces** of the data blocks and uses singular value decomposition (SVD), 
principal-angle ideas, and matrix perturbation theory to identify variation shared 
across blocks.

For data blocks

$$
X_k \in \mathbb{R}^{p_k \times n}, \qquad k=1,\ldots,K,
$$

the rows correspond to features and the columns correspond to the same $n$ samples in every block. AJIVE models each block as

$$
X_k = J_k + I_k + E_k,
$$

where

- $J_k$ is the variation that is joint across all blocks,
- $I_k$ is structured variation specific to block $k$,
- $E_k$ is residual variation.

The joint components share a common score subspace,

$$
\operatorname{row}(J_1)
=
\cdots
=
\operatorname{row}(J_K),
$$

while each individual score subspace is orthogonal to the joint score subspace.

AJIVE proceeds in three main steps.



### Step 1: Initial signal-space extraction

Each data block is first approximated by a low-rank SVD.

For block $k$,

$$
X_k = U_k \Sigma_k V_k^\top.
$$

An initial signal rank $\tilde r_k$ is selected, and only the leading $\tilde r_k$ singular components are retained:

$$
\widetilde A_k
=
\widetilde U_k
\widetilde \Sigma_k
\widetilde V_k^\top.
$$

The columns of 
$
\widetilde V_k \in \mathbb{R}^{n \times \tilde r_k}
$
form an orthonormal basis for the estimated signal score subspace of block $k$.

The initial ranks determine how much variation is treated as signal before joint 
and individual variation are separated. If an initial rank is too small, joint signal 
may be omitted. If it is too large, individual variation or noise may enter the 
subsequent joint-space calculation.

When initial ranks are supplied directly, a convenient singular-value threshold for 
block $k$ is

$$
t_k
=
\frac{\sigma_{\tilde r_k}+\sigma_{\tilde r_k+1}}{2},
$$

where $\sigma_j$ denotes the $j$-th singular value of $X_k$.

### Step 2: Score-space segmentation

AJIVE identifies joint variation by comparing the estimated score subspaces 
from Step 1.

The score bases are stacked vertically:

$$
M =
\begin{bmatrix}
\widetilde V_1^\top \\
\widetilde V_2^\top \\
\vdots \\
\widetilde V_K^\top
\end{bmatrix}.
$$

An SVD is then computed:

$$
M = U_M \Sigma_M V_M^\top.
$$

The singular values of $M$ measure how strongly a direction is represented across the block-specific signal subspaces. For $K$ blocks, a squared singular value close to $K$ indicates a direction that is strongly shared across the blocks.

The estimated joint score basis is formed from the leading right singular vectors 
of $M$:

$$
V_J = V_M[:,1:r_J],
$$

where $r_J$ is the estimated joint rank.

In the full AJIVE procedure, $r_J$ is determined using two complementary criteria:

1. **Random-direction bound.** Random subspaces with the same dimensions as the 
estimated signal spaces are generated. Their largest squared singular values provide 
a reference distribution for alignments that could occur by chance.
2. **Wedin perturbation bound.** Matrix perturbation theory is used to estimate how 
much the signal subspaces could have moved because of noise. This gives a second 
threshold for deciding whether an observed shared direction is consistent with a 
perturbed version of a true joint component.

A component is classified as joint only when it satisfies both criteria.

### Step 3: Final decomposition and outputs

After the joint score basis $V_J$ has been identified, each data block is projected onto the joint score subspace.

The joint component of block $k$ is

$$
\widehat J_k
=
X_k V_J V_J^\top.
$$

In an implementation, the projection matrix $V_JV_J^\top$ does not need to be formed 
explicitly. The equivalent computation

$$
\widehat J_k
=
(X_kV_J)V_J^\top
$$

is more memory-efficient.

Before accepting a candidate joint component, AJIVE checks that it is sufficiently 
strong in every block. For candidate joint direction $v_j$,

$$
\|X_k v_j\|_2 > t_k
$$

must hold for each block. Components that fail this check are removed from the final \joint basis.

The joint variation is then removed:

$$
X_k^\perp
=
X_k - \widehat J_k.
$$

An SVD of $X_k^\perp$ is computed, and singular components above the Step 1 
threshold $t_k$ are retained as individual structure:

$$
\widehat I_k.
$$

The remaining variation is assigned to the residual:

$$
\widehat E_k
=
X_k
-
\widehat J_k
-
\widehat I_k.
$$

The final AJIVE decomposition is therefore

$$
\boxed{
X_k
=
\widehat J_k
+
\widehat I_k
+
\widehat E_k
}.
$$

---

## Summary

AJIVE can be summarized as:

1. **Extract low-rank signal spaces** from each block using SVD.
2. **Identify the common score subspace** by comparing the block-specific score 
spaces.
3. **Project each block** onto the joint score subspace and its orthogonal complement 
to obtain joint, individual, and residual variation.

The key distinction between AJIVE and the original JIVE algorithm is that AJIVE works 
directly with **score subspaces** rather than iteratively estimating the full joint 
and individual matrices.

---

## Reference

Feng, Q., Jiang, M., Hannig, J., & Marron, J. S. (2018).
*Angle-based joint and individual variation explained.* Journal of Multivariate Analysis, 166, 241–265. https://doi.org/10.1016/j.jmva.2018.03.008
