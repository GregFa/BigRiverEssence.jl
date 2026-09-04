"""
    Ajive{T}

Container for a fitted AJIVE (Angle-based Joint and Individual Variation
Explained) decomposition, as returned by [`ajive`](@ref).

AJIVE uses a canonical internal orientation of features × samples (`pᵢ × n`),
regardless of the orientation supplied by the caller.

# Fields
- `J::Vector{Matrix{T}}`: Joint structure, one `pᵢ×n` matrix per block.
- `I::Vector{Matrix{T}}`: Individual structure, one `pᵢ×n` matrix per block.
- `E::Vector{Matrix{T}}`: Residual/noise structure, one `pᵢ×n` matrix per block.
- `S::Matrix{T}`: `r×n` common normalized joint scores (orthonormal rows).
- `U::Vector{Matrix{T}}`: Joint block coefficients/loadings, one `pᵢ×r`
  matrix per block, so that `J[i] = U[i] * S`.
- `Si::Vector{Matrix{T}}`: Individual scores, one `rᵢ×n` matrix per block.
- `Wi::Vector{Matrix{T}}`: Individual loadings, one `pᵢ×rᵢ` matrix per block,
  so that `I[i] = Wi[i] * Si[i]`.
- `init_ranks::Vector{Int}`: Initial signal ranks used in AJIVE Step 1.
- `requested_r::Int`: Candidate joint rank supplied to Step 2.
- `r::Int`: Final joint rank after the optional Step-3 identifiability check.
- `ri::Vector{Int}`: Final individual ranks, one per block.
- `thresholds::Vector{T}`: Step-1 singular-value thresholds, one per block.
- `init_svals::Vector{Vector{T}}`: Leading `init_rank + 1` singular values used
  to define each threshold.
- `common_svals::Vector{T}`: Singular values from the stacked score-space SVD,
  through the maximum possible fully joint rank `minimum(init_ranks)`.
- `dropped_joint::Vector{Int}`: Candidate joint-component indices removed by the
  Step-3 identifiability check.
- `means::Vector{Vector{T}}`: Feature means removed from each canonical `pᵢ×n`
  block (zeros when `center=false`).
- `centered::Bool`: Whether feature centering was performed.
- `orientation::Symbol`: Orientation of the caller's input matrices.
"""
struct Ajive{T}
    J::Vector{Matrix{T}}
    I::Vector{Matrix{T}}
    E::Vector{Matrix{T}}
    S::Matrix{T}
    U::Vector{Matrix{T}}
    Si::Vector{Matrix{T}}
    Wi::Vector{Matrix{T}}
    init_ranks::Vector{Int}
    requested_r::Int
    r::Int
    ri::Vector{Int}
    thresholds::Vector{T}
    init_svals::Vector{Vector{T}}
    common_svals::Vector{T}
    dropped_joint::Vector{Int}
    means::Vector{Vector{T}}
    centered::Bool
    orientation::Symbol
end
