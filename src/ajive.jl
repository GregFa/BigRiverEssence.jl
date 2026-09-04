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

"""
    _ajive_prepare_blocks(Xs::AbstractVector{<:AbstractMatrix{<:Real}};
        orientation::Symbol = :features_by_samples,
        center::Bool = true)

Prepare the data blocks for AJIVE by centering across samples if requested, converting to Float64, and 
ensuring consistent orientation, (features × samples (`pᵢ×n`)).  

# Arguments
- `Xs`: Vector of data blocks, each a matrix of real numbers.
- `orientation`: Symbol indicating the orientation of the input matrices (`:features_by_samples` or `:samples_by_features`).
- `center`: Boolean indicating whether to center the features.

# Value 
- A tuple `(Xc, means, n)` where `Xc` is a vector of centered Float64 matrices in 
canonical orientation (features × samples (`pᵢ×n`)), 
`means` is a vector of feature means for each block, and `n` is the number of samples.
"""
function _ajive_prepare_blocks(Xs::AbstractVector{<:AbstractMatrix{<:Real}};
    orientation::Symbol = :features_by_samples,
    center::Bool = true)

    length(Xs) >= 2 || throw(ArgumentError("AJIVE requires at least two data blocks"))
    orientation in (:features_by_samples, :samples_by_features) ||
        throw(ArgumentError("orientation must be :features_by_samples or :samples_by_features"))

    # Prepare the data blocks: center if requested, ensure Float64 type, and check dimensions
    k = length(Xs)
    Xc = Vector{Matrix{Float64}}(undef, k)
    means = Vector{Vector{Float64}}(undef, k)

    # Convert each block to the canonical orientation and type, and center if requested    
    for b in 1:k
        X = Xs[b]
        Y = if orientation === :features_by_samples
            Float64.(X)
        else
            # Materialize the caller's n×p block directly into our canonical
            # p×n Float64 layout in one allocation.
            Float64.(permutedims(X))
        end

        size(Y, 1) > 0 && size(Y, 2) > 0 ||
            throw(ArgumentError("AJIVE data blocks must be non-empty"))

        if center
            μ = mean(Y, dims = 2)
            Y .-= μ
            means[b] = vec(μ)
        else
            means[b] = zeros(Float64, size(Y, 1))
        end
        Xc[b] = Y
    end

    n = size(Xc[1], 2)
    all(size(X, 2) == n for X in Xc) ||
        throw(ArgumentError("all AJIVE blocks must contain the same number of samples"))

    return Xc, means, n
end