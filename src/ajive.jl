########################
# Structures for AJIVE #
########################
"""
    AjiveRankDiagnostics{T}
    
    A struct to hold the diagnostics for the rank selection in AJIVE.

# Fields
- `random_threshold::T`: The threshold computed from the random-direction bound. It 
is the quantile of the largest singular values squared of random matrices with the 
same dimensions as the data blocks.
- `wedin_threshold::T`: The threshold computed from the Wedin bound. It is 
the quantile of the largest singular values squared of the Wedin matrices for 
each block.
- `combined_threshold::T`: The combined threshold. It is used for joint-rank selection 
and is the maximum of the random-direction and Wedin thresholds.
- `random_samples::Vector{T}`: The simulated largest squared sigular values from rendom
subspaces of the random-direction bound.
- `wedin_samples::Vector{T}`: The simulated multi-blocks Wedin lower-bound values.
- `block_wedin_samples::Vector{Vector{T}}`: The block-specific samples of the estimated 
`sin(θ)` pertubation bound.

"""
struct AjiveRankDiagnostics{T}
    random_threshold::T
    wedin_threshold::T
    combined_threshold::T
    random_samples::Vector{T}
    wedin_samples::Vector{T}
    block_wedin_samples::Vector{Vector{T}}
end


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
    rank_diagnostics::Union{Nothing,AjiveRankDiagnostics{T}}
    means::Vector{Vector{T}}
    centered::Bool
    orientation::Symbol
end


#########
# AJIVE # 
#########


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

    # Validate the input parameters for preparing the data blocks
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


"""

    _ajive_check_ranks(Xs, init_ranks, joint_rank, indiv_ranks = nothing)

Check the validity of the initial, joint, and individual ranks for AJIVE data blocks.

# Arguments
- `Xs`: A vector of Float64 matrices representing the AJIVE data blocks.
- `init_ranks`: A vector of initial signal ranks for each block.
- `joint_rank`: The joint rank across all blocks.
- `indiv_ranks`: (Optional) A vector of individual ranks for each block.

# Throws
- `ArgumentError` if any of the ranks are out of their valid ranges.
"""
function _ajive_check_ranks(Xs::Vector{Matrix{Float64}}, init_ranks::Vector{Int}, 
    joint_rank::Int, indiv_ranks::Union{Nothing,Vector{Int}} = nothing)

    k = length(Xs)
    length(init_ranks) == k ||
        throw(ArgumentError("one initial signal rank is required for each AJIVE block"))

    # Validate that the initial signal ranks are within the valid range for each block    
    for b in 1:k
        # The maximum initial rank is one less than the minimum dimension of the block
        max_init = min(size(Xs[b])...) - 1
        # Validate that the initial rank is between 1 and max_init
        (1 <= init_ranks[b] <= max_init) ||
            throw(ArgumentError("initial signal rank for block $b must be between 1 and $max_init"))
    end

    # Validate that the joint rank is within the valid range
    (0 <= joint_rank <= minimum(init_ranks)) ||
        throw(ArgumentError("joint_rank must be between 0 and minimum(init_ranks)"))

    # Validate the individual ranks if provided
    if indiv_ranks !== nothing
        length(indiv_ranks) == k ||
            throw(ArgumentError("one individual rank is required for each AJIVE block"))
        for b in 1:k
            # The maximum individual rank is the minimum dimension of the block
            max_indiv = min(size(Xs[b])...)
            (0 <= indiv_ranks[b] <= max_indiv) ||
                throw(ArgumentError("individual rank for block $b must be between 0 and $max_indiv"))
        end
    end
    return nothing # All rank validations passed
end


"""
    _ajive_initial_signal_svd(Xs::Vector{Matrix{Float64}}, init_ranks::Vector{Int})

Compute the initial signal SVD for each AJIVE data block. AJIVE Step 2 with a fixed joint rank: 
vertically stack the transposed initial score-space bases and compute its SVD. 
Returns a basis for the candidate common score directions and the corresponding singular values.

# Arguments
- `Xs`: A vector of Float64 matrices representing the AJIVE data blocks.
- `init_ranks`: A vector of initial signal ranks for each block.

# Values
- `Us`: A vector of matrices containing the left singular vectors for each block.
- `svals`: A vector of vectors containing the singular values for each block.
- `Vs`: A vector of matrices containing the right singular vectors for each block.
- `thresholds`: A vector of thresholds for separating signal from noise for each block.
- `init_svals`: A vector of vectors containing the initial singular values for each block.
"""    
function _ajive_initial_signal_svd(Xs::Vector{Matrix{Float64}}, init_ranks::Vector{Int})

    # Initialize the arrays for the initial signal SVD results
    k = length(Xs)
    Us = Vector{Matrix{Float64}}(undef, k)
    svals = Vector{Vector{Float64}}(undef, k)
    Vs = Vector{Matrix{Float64}}(undef, k)
    thresholds = Vector{Float64}(undef, k)
    init_svals = Vector{Vector{Float64}}(undef, k)

    # Compute the initial signal SVD for each block
    for b in 1:k
        r = init_ranks[b]
        F = _safe_svd(Xs[b])

        # Extract the initial signal components from the SVD result
        init_svals[b] = Vector{Float64}(@view F.S[1:(r + 1)])
        
        # Compute the threshold for separating signal from noise (based on the original implementation of AJIVE)
        thresholds[b] = (F.S[r] + F.S[r + 1]) / 2
        
        # Extract the left singular vectors corresponding to the initial signal
        Us[b] = Matrix{Float64}(@view F.U[:, 1:r])
        
        # Extract the singular values corresponding to the initial signal
        svals[b] = Vector{Float64}(@view F.S[1:r])
        
        # Extract the right singular vectors corresponding to the initial signal
        Vs[b] = Matrix{Float64}(transpose(@view F.Vt[1:r, :]))
    end

    return Us, svals, Vs, thresholds, init_svals
end


"""
    ajive_common_score_space(Vs::Vector{Matrix{Float64}}, init_ranks::Vector{Int}, 
        n::Int)

Compute the common score space from the right singular vectors of each block.

# Arguments
- `Vs`: A vector of matrices containing the right singular vectors for each block.
- `init_ranks`: A vector of initial signal ranks for each block.
- `n`: The number of samples (columns) in each block.   

# Values
- `Vcommon`: A matrix containing the left singular vectors corresponding to the common score space.
- `common_svals`: A vector containing the singular values corresponding to the common score space.      

"""
function _ajive_common_score_space(Vs::Vector{Matrix{Float64}}, init_ranks::Vector{Int}, n::Int)

    # Compute the common score space by vertically stacking the transposed 
    # right singular vectors and computing their SVD
    total_rank = sum(init_ranks)
    M = Matrix{Float64}(undef, total_rank, n)
    firstrow = 1
    
    # Fill the stacked matrix M with the transposed right singular vectors from each block
    # This creates a combined matrix of all right singular vectors across blocks    
    for b in eachindex(Vs)
        r = init_ranks[b]
        rows = firstrow:(firstrow + r - 1)
        @views M[rows, :] .= transpose(Vs[b])
        firstrow += r
    end

    # Compute the SVD of the stacked matrix to find the common score space
    F = _safe_svd(M)
    max_joint_rank = minimum(init_ranks)
    
    # Extract the left singular vectors corresponding to the maximum possible joint rank
    Vcommon = Matrix{Float64}(transpose(@view F.Vt[1:max_joint_rank, :]))
    
    # Extract the singular values corresponding to the maximum possible joint rank
    common_svals = Vector{Float64}(@view F.S[1:max_joint_rank])

    return Vcommon, common_svals
end

################################
# AJIVE random-direction bound #
################################
"""
    ajive_random_orthonormal(n::Int, r::Int, rng::AbstractRNG)
Generate a random orthonormal matrix of size `n×r` using QR decomposition.

# Arguments
- `n`: The number of rows in the matrix.
- `r`: The number of columns in the matrix.
- `rng`: The random number generator to use.

# Values
- `Q`: A random orthonormal matrix of size `n×r`.   

*Notes:  A Gaussian matrix followed by QR is sufficient because AJIVE uses only
the subspace spanned by the columns of Q, not the specific orthonormal basis.  The
random-direction bound is used in AJIVE's Step 1 to estimate the largest 
singular value of a random matrix with the same dimensions as the data block. 
The bound is used to determine whether a candidate joint direction is strong enough 
to be considered signal rather than noise.
"""
function _ajive_random_orthonormal(n::Int, r::Int, rng::AbstractRNG)
    # Validate that the random subspace dimension is within the valid range
    1 <= r <= n || throw(ArgumentError("random subspace dimension must satisfy 1 ≤ r ≤ n"))
    
    # Generate a random n×r matrix with standard normal entries
    Z = randn(rng, n, r)

    # Perform QR decomposition to obtain an orthonormal basis for the random subspace
    F = qr!(Z)
    
    return Matrix{Float64}(F.Q[:, 1:r])
end

"""
    _ajive_largest_svalsq(M::Matrix{Float64})

Compute the largest singular value squared of the matrix M.

# Arguments
- `M`: A matrix.

# Values
- The largest singular value squared of the matrix M.

*Notes: The computation uses the smaller Gram matrix instead of a full SVD, 
which is useful because this quantity is evaluated repeatedly when simulating 
the random-direction bound.*
"""
function _ajive_largest_svalsq(M::Matrix{Float64})
    m, n = size(M)

    # Compute the largest singular value squared of the matrix M using 
    # the eigenvalues of M*M' or M'*M, depending on the dimensions of M.
    if m <= n
        G = Symmetric(M * transpose(M))
    else
        G = Symmetric(transpose(M) * M)
    end

    return eigmax(G)
end

"""
    _ajive_random_direction_samples(n::Int, dims::Vector{Int};
        n_samples::Int = 1000,
        rng::AbstractRNG = Random.default_rng())
        
Generate samples of the largest singular value squared of a random matrix
with the given dimensions.

# Arguments
- `n`: The ambient dimension of the random matrix.
- `dims`: A vector of dimensions for each random subspace.
- `n_samples`: The number of samples to generate.
- `rng`: The random number generator to use.

# Values
- A vector of the largest singular value squared for each sample.

*Notes: This function is used to simulate the random-direction bound in AJIVE.*
"""
function _ajive_random_direction_samples(n::Int, dims::Vector{Int};
    n_samples::Int = 1000,
    rng::AbstractRNG = Random.default_rng())

    # Validate the input parameters for generating random direction samples
    n > 0 || throw(ArgumentError("ambient sample-space dimension n must be positive"))
    length(dims) >= 2 || throw(ArgumentError("random-direction bound requires at least two subspaces"))
    n_samples > 0 || throw(ArgumentError("n_samples must be positive"))
    
    # Validate that each random subspace dimension is between 1 and n
    all(r -> 1 <= r <= n, dims) ||
        throw(ArgumentError("every random subspace dimension must be between 1 and n"))

    # Compute the total rank of the stacked random subspaces
    total_rank = sum(dims)

    # Initialize the matrix M to hold the stacked random subspaces and 
    # a vector to hold the largest singular value squared for each sample
    M = Matrix{Float64}(undef, total_rank, n)

    # Initialize a vector to hold the largest singular value squared 
    # for each sample
    samples = Vector{Float64}(undef, n_samples)

    # Generate random direction samples by creating random orthonormal subspaces
    for s in 1:n_samples
        firstrow = 1
        for r in dims
            rows = firstrow:(firstrow + r - 1)
            
            # Generate a random orthonormal matrix of size n×r 
            # and fill the corresponding rows of M
            Q = _ajive_random_orthonormal(n, r, rng)
            
            # Fill the corresponding rows of M with the transpose of Q
            @views M[rows, :] .= transpose(Q)
            firstrow += r
        end
        samples[s] = _ajive_largest_svalsq(M)
    end

    return samples
end


"""
    _ajive_random_direction_bound(n::Int, dims::Vector{Int};
    n_samples::Int = 1000,
    percentile::Real = 0.95,
    rng::AbstractRNG = Random.default_rng())

Compute the random-direction bound for AJIVE by generating samples of 
the largest singular value squared of a random matrix with the given 
dimensions and computing the requested quantile. 

# Arguments
- `n`: The ambient dimension of the random matrix.
- `dims`: A vector of dimensions for each random subspace.
- `n_samples`: The number of samples to generate.       
- `percentile`: The quantile to compute for the random-direction bound.
- `rng`: The random number generator to use.

# Values
- `threshold`: The computed random-direction bound (quantile of the samples).
- `samples`: A vector of the largest singular value squared for each sample.    

*Notes: This function is used to compute the random-direction bound for AJIVE.*
"""
function _ajive_random_direction_bound(n::Int, dims::Vector{Int};
    n_samples::Int = 1000,
    percentile::Real = 0.95,
    rng::AbstractRNG = Random.default_rng())

    # Validate the input parameters for generating the random-direction bound
    0 < percentile < 1 || throw(ArgumentError("percentile must lie strictly between 0 and 1"))
    
    # Generate random direction samples and compute the requested 
    # quantile as the threshold
    samples = _ajive_random_direction_samples(n, dims;
                n_samples = n_samples, rng = rng)
    # Compute the requested quantile of the samples to 
    # determine the threshold
    threshold = quantile(samples, Float64(percentile))
    
    return threshold, samples
end

#####################################################################
# AJIVE Wedin perturbation bound and automatic joint-rank selection #
#####################################################################
"""
    _ajive_random_orthogonal_subspace(basis::Matrix{Float64}, r::Int,
        rng::AbstractRNG)

Draw a random `r`-dimensional orthonormal subspace that is orthogonal to the
columns of `basis`. This is the resampling device used by AJIVE to estimate the
unknown projected-noise terms in Wedin's sin(θ) perturbation bound.

A true `r`-dimensional orthogonal subspace exists only when the orthogonal
complement has dimension at least `r`.

# Arguments
- `basis`: A matrix whose columns define the signal subspace.
- `r`: The dimension of the desired orthogonal subspace.
- `rng`: The random number generator to use.

# Values
- `Q`: A matrix whose columns form an orthonormal basis for the orthogonal subspace.

"""
function _ajive_random_orthogonal_subspace(basis::Matrix{Float64}, r::Int,
    rng::AbstractRNG)

    dim, rbasis = size(basis)

    # Validate the input basis and requested subspace dimension.
    rbasis == r || throw(ArgumentError("basis must have exactly r columns"))
    r > 0 || throw(ArgumentError("r must be positive"))
    dim - r >= r || throw(ArgumentError(
        "Wedin resampling needs an r-dimensional subspace orthogonal to a rank-r basis; " *
        "require ambient dimension ≥ 2r (got dimension=$dim, r=$r)"))

    # Generate a random matrix with the specified dimensions.
    Z = randn(rng, dim, r)
    
    # Project random directions off the estimated signal subspace.
    Z .-= basis * (transpose(basis) * Z)
    
    # Perform a QR decomposition of the resulting matrix.
    F = qr!(Z)
    
    return Matrix{Float64}(F.Q[:, 1:r])
end


"""
    _ajive_wedin_samples(X::Matrix{Float64}, U::Matrix{Float64},
    svals::Vector{Float64}, V::Matrix{Float64};
    n_samples::Int = 1000, rng::AbstractRNG = Random.default_rng()) 

Estimate the block-specific Wedin perturbation distribution for one AJIVE data
block in the paper's features x samples orientation.

For a rank-`r` initial signal approximation `U * Diagonal(svals) * V'`, each
replicate independently samples

- an `r`-dimensional subspace `Vstar` orthogonal to `V`, and
- an `r`-dimensional subspace `Ustar` orthogonal to `U`,

then evaluates

`min(max(opnorm(X * Vstar), opnorm(X' * Ustar)) / svals[end], 1)`.

These samples estimate the unknown `sin(theta)` term in Wedin's bound.

# Arguments 
- `X`: The data block matrix.
- `U`: The left singular vectors of the initial signal approximation.
- `svals`: The singular values of the initial signal approximation.
- `V`: The right singular vectors of the initial signal approximation.
- `n_samples`: The number of samples to generate.
- `rng`: The random number generator to use.

# Values
- A vector of estimated `sin(theta)` values for the Wedin perturbation bound.

"""
function _ajive_wedin_samples(X::Matrix{Float64}, U::Matrix{Float64},
    svals::Vector{Float64}, V::Matrix{Float64};
    n_samples::Int = 1000, rng::AbstractRNG = Random.default_rng())

    # Validate the input parameters for generating Wedin samples
    n_samples > 0 || throw(ArgumentError("n_samples must be positive"))
    r = length(svals)
    r > 0 || throw(ArgumentError("Wedin resampling requires a positive signal rank"))
    size(U, 2) == r || throw(ArgumentError("U must have one column per retained singular value"))
    size(V, 2) == r || throw(ArgumentError("V must have one column per retained singular value"))
    size(U, 1) == size(X, 1) || throw(DimensionMismatch("U is incompatible with X"))
    size(V, 1) == size(X, 2) || throw(DimensionMismatch("V is incompatible with X"))

    # Validate complement dimensions once, before entering the Monte Carlo loop.
    size(U, 1) - r >= r || throw(ArgumentError(
        "Wedin resampling for the feature-side signal basis requires p ≥ 2r"))
    size(V, 1) - r >= r || throw(ArgumentError(
        "Wedin resampling for the sample-side signal basis requires n ≥ 2r"))

    # Compute the smallest retained singular value, 
    # which is used to normalize the perturbation ratio.    
    sigma_min = svals[end]
    samples = Vector{Float64}(undef, n_samples)

    # A zero weakest retained singular value makes the perturbation ratio
    # unbounded; clipping by one gives the maximally conservative Wedin value.
    if sigma_min <= eps(Float64)
        fill!(samples, 1.0)
        return samples
    end

    # Generate Wedin samples by drawing random orthogonal 
    # subspaces and computing the perturbation ratio.
    for s in 1:n_samples
        Vstar = _ajive_random_orthogonal_subspace(V, r, rng)
        Ustar = _ajive_random_orthogonal_subspace(U, r, rng)

        # Compute the noise levels in the right and left directions.
        # This is done by computing the operator norm of the product 
        # of the data matrix and the random subspace.
        right_noise = opnorm(X * Vstar)
        left_noise = opnorm(transpose(X) * Ustar)
        samples[s] = min(max(right_noise, left_noise) / sigma_min, 1.0)
    end

    return samples
end



"""
    _ajive_wedin_bound(Xs::Vector{Matrix{Float64}},
        Us::Vector{Matrix{Float64}}, svals::Vector{Vector{Float64}},
        Vs::Vector{Matrix{Float64}};
        n_samples::Int = 1000,
        percentile::Real = 0.05,
        rng::AbstractRNG = Random.default_rng())

Estimate the AJIVE joint rank by combining the random-direction and Wedin
bounds. The effective squared-singular-value cutoff is the larger of the two
bounds, and the candidate joint rank is the number of common-space squared
singular values above that cutoff.

# Arguments
- `Xs`: A vector of matrices containing the data for each block.
- `Us`: A vector of matrices containing the left singular vectors for each block.
- `svals`: A vector of vectors containing the singular values for each block.
- `Vs`: A vector of matrices containing the right singular vectors for each block.
- `n_samples`: The number of samples to generate for the Wedin bound.
- `percentile`: The quantile to compute for the Wedin bound.
- `rng`: The random number generator to use.    

# Values
- `threshold`: The computed Wedin bound (quantile of the samples).
- `samples`: A vector of the Wedin perturbation estimates for each sample.
- `block_samples`: A vector of vectors containing the block-specific Wedin 
    samples for each block

"""
function _ajive_wedin_bound(Xs::Vector{Matrix{Float64}},
    Us::Vector{Matrix{Float64}}, svals::Vector{Vector{Float64}},
    Vs::Vector{Matrix{Float64}};
    n_samples::Int = 1000,
    percentile::Real = 0.05,
    rng::AbstractRNG = Random.default_rng())

    # Validate the input parameters for generating the Wedin bound
    k = length(Xs)

    # The Wedin bound requires at least two data blocks to compute 
    # the perturbation distribution across blocks.
    k >= 2 || throw(ArgumentError("Wedin bound requires at least two data blocks"))
    length(Us) == k && length(svals) == k && length(Vs) == k ||
        throw(ArgumentError("Xs, Us, svals, and Vs must contain the same number of blocks"))
    n_samples > 0 || throw(ArgumentError("n_samples must be positive"))
    0 < percentile < 1 || throw(ArgumentError("percentile must lie strictly between 0 and 1"))

    # Compute the block-specific Wedin samples for each data block
    block_samples = Vector{Vector{Float64}}(undef, k)

    # Compute the block-specific Wedin samples for each data block
    for b in 1:k
        block_samples[b] = _ajive_wedin_samples(
            Xs[b], Us[b], svals[b], Vs[b]; n_samples = n_samples, rng = rng)
    end

    # Compute the combined Wedin samples by subtracting the squared block samples
    samples = fill(Float64(k), n_samples)
    for b in 1:k
        @. samples -= block_samples[b]^2
    end

    # Compute the threshold for the combined Wedin samples
    threshold = quantile(samples, Float64(percentile))


    return threshold, samples, block_samples
end


"""
    _ajive_check_identifiability(Xs::Vector{Matrix{Float64}},
    VJ::Matrix{Float64}, thresholds::Vector{Float64})    

Check the identifiability of the joint score space. A candidate joint 
direction is retained only when its projection has singular-value 
magnitude at least the Step-1 threshold in every block. Identifiable 
means a candidate joint direction is supported by every data block 
strongly enough to be considered real signal rather than noise.

# Arguments
- `Xs`: A vector of matrices containing the data for each block.
- `VJ`: A matrix containing the left singular vectors corresponding to the joint score space.
- `thresholds`: A vector of thresholds for each block.

# Values
- `VJ`: A matrix containing the left singular vectors corresponding to the identifiable joint score space.
- `dropped`: A vector containing the indices of the dropped components.

"""
function _ajive_check_identifiability(Xs::Vector{Matrix{Float64}},
    VJ::Matrix{Float64}, thresholds::Vector{Float64})

    r = size(VJ, 2)
    r == 0 && return VJ, Int[]

    # A component is identifiable when every block's projection meets its threshold
    identifiable = [all(norm(Xs[b] * view(VJ, :, j)) >= thresholds[b] for b in eachindex(Xs)) for j in 1:r]
    dropped = findall(!, identifiable)

    return Matrix{Float64}(VJ[:, identifiable]), dropped
end

"""
_ajive_final_decomposition(Xs::Vector{Matrix{Float64}},
    VJ::Matrix{Float64}, thresholds::Vector{Float64};
    indiv_ranks::Union{Nothing,Vector{Int}} = nothing)

Perform the final decomposition of the AJIVE data blocks into 
joint, individual, and residual components. Reconstruct the joint 
components using the joint score space and compute 
the individual components by projecting the residuals 
onto the individual score spaces. 
The residuals are computed as the difference between 
the original data and the sum of the joint and individual 
components.

# Arguments
- `Xs`: A vector of matrices containing the data for each block.
- `VJ`: A matrix containing the left singular vectors corresponding to the joint score space.
- `thresholds`: A vector of thresholds for each block.
- `indiv_ranks`: (Optional) A vector of individual ranks for each 
    block. When not provided, the individual ranks are determined 
    by the number of singular values of the joint-orthogonal block 
    above the thresholds.

# Values
- `J`: A vector of matrices containing the joint components for 
    each block.
- `Iblocks`: A vector of matrices containing the individual 
    components for each block.
- `E`: A vector of matrices containing the residual components 
    for each block.
- `U`: A vector of matrices containing the joint loadings for 
    each block.   
- `Si`: A vector of matrices containing the individual scores 
    for each block.
- `Wi`: A vector of matrices containing the individual loadings 
    for each block.
- `ri`: A vector of integers containing the individual ranks 
    for each block.
"""
function _ajive_final_decomposition(Xs::Vector{Matrix{Float64}},
    VJ::Matrix{Float64}, thresholds::Vector{Float64};
    indiv_ranks::Union{Nothing,Vector{Int}} = nothing)

    k = length(Xs)
    n = size(Xs[1], 2)
    r = size(VJ, 2)

    # Initialize the arrays for the final decomposition results
    J = Vector{Matrix{Float64}}(undef, k)
    Iblocks = Vector{Matrix{Float64}}(undef, k)
    E = Vector{Matrix{Float64}}(undef, k)
    U = Vector{Matrix{Float64}}(undef, k)
    Si = Vector{Matrix{Float64}}(undef, k)
    Wi = Vector{Matrix{Float64}}(undef, k)
    ri = Vector{Int}(undef, k)

    # Compute the joint, individual, and 
    # residual components for each block
    for b in 1:k
        X = Xs[b]
        p = size(X, 1)

        # Compute the joint component for the current block
        if r == 0
            U[b] = zeros(Float64, p, 0)
            J[b] = zeros(Float64, p, n)
            Xorth = copy(X)
        else
            U[b] = X * VJ
            J[b] = U[b] * transpose(VJ)
            Xorth = X - J[b]
        end

        # Compute the individual components for the current 
        # block
        F = _safe_svd(Xorth)
        # Determine the individual rank for the current block,
        # either from the provided indiv_ranks or by counting
        # the number of singular values above the threshold
        rb = indiv_ranks === nothing ? count(>(thresholds[b]), F.S) : indiv_ranks[b]
        max_indiv = min(p, n - r)
        0 <= rb <= max_indiv ||
            throw(ArgumentError("individual rank for block $b must be between 0 and $max_indiv after the final joint-rank check"))
        ri[b] = rb

        # Compute the individual components for the current block
        if rb == 0
            Si[b] = zeros(Float64, 0, n)
            Wi[b] = zeros(Float64, p, 0)
            Iblocks[b] = zeros(Float64, p, n)
        else
            Si[b] = Matrix{Float64}(@view F.Vt[1:rb, :])
            Wi[b] = Matrix{Float64}(@view F.U[:, 1:rb]) * Diagonal(@view F.S[1:rb])
            Iblocks[b] = Wi[b] * Si[b]
        end

        # Compute the residuals for the current block
        E[b] = X - J[b] - Iblocks[b]
    end

    return J, Iblocks, E, U, Si, Wi, ri
end

"""

    ajive(Xs::AbstractVector{<:AbstractMatrix{<:Real}};
        init_ranks::Vector{Int},
        joint_rank::Union{Nothing,Int,Symbol} = :auto,
        indiv_ranks::Union{Nothing,Vector{Int}} = nothing,
        orientation::Symbol = :features_by_samples,
        center::Bool = true,
        check_joint_identifiability::Bool = true,
        n_rand_samples::Int = 1000,
        rand_percentile::Real = 0.95,
        n_wedin_samples::Int = 1000,
        wedin_percentile::Real = 0.05,
        rng::AbstractRNG = Random.default_rng())


Fit Angle-based Joint and Individual Variation Explained (AJIVE).

This implementation follows the three-step AJIVE construction of Feng, Jiang,
Hannig & Marron (2018) using the paper's features × samples convention
(`pᵢ×n`) internally. When `joint_rank` is `:auto` (default) or `nothing`, Step 2
estimates it from the random-direction and Wedin bounds. Supplying an integer
keeps the deterministic fixed-rank behavior from Milestone 1.

# Arguments
- `Xs`: At least two data blocks containing the same samples in the same order.
- `init_ranks::Vector{Int}`: Initial signal rank for each block. Because the
  associated Step-1 threshold uses `σᵣ` and `σᵣ₊₁`, each rank must be at most
  `min(pᵢ,n)-1`.
- `joint_rank`: Candidate joint rank. When `nothing` or `:auto` (default),
  estimate the rank from the random-direction and Wedin bounds.
- `indiv_ranks`: Optional individual ranks. When omitted, Step 3 estimates each
  rank by retaining singular values larger than the corresponding Step-1
  threshold.
- `orientation::Symbol`: `:features_by_samples` (default) for `pᵢ×n` input, or
  `:samples_by_features` for `n×pᵢ` input. The latter is materialized once as
  `pᵢ×n` before the AJIVE computation.
- `center::Bool`: Whether to center each feature across samples. Defaults to true.
- `check_joint_identifiability::Bool`: Whether to apply AJIVE's Step-3 check that
  every candidate joint direction remains above the Step-1 signal threshold in
  every block. Defaults to true.
- `n_rand_samples::Int`: Number of Monte Carlo samples for the random-direction
  bound when estimating `joint_rank`. Defaults to 1000.
- `rand_percentile::Real`: Upper quantile of the random-direction squared
  singular-value distribution. Defaults to 0.95.
- `n_wedin_samples::Int`: Number of Monte Carlo samples for the Wedin bound when
  estimating `joint_rank`. Defaults to 1000.
- `wedin_percentile::Real`: Lower quantile of the multi-block Wedin lower-bound
  distribution. Defaults to 0.05.
- `rng::AbstractRNG`: Random-number generator used by the automatic rank
  estimation. Pass a seeded RNG for reproducibility.

# Value
An [`Ajive`](@ref) object. `J`, `I`, and `E` are always returned in the canonical
features × samples orientation and decompose the centered canonical blocks:
`X_centered = J + I + E`.

When the joint rank is estimated automatically, `m.rank_diagnostics` contains
both simulated null distributions and their thresholds. When an integer joint
rank is supplied, `m.rank_diagnostics === nothing`.

# Notes
The common normalized scores are stored in `m.S` with shape `r×n`. Their
transpose is the orthonormal basis `V_J` of the estimated joint score subspace.
The implementation avoids explicitly forming the `n×n` projector
`V_J * V_J'`; instead each joint block is computed as `(X * V_J) * V_J'`.

The Wedin resampling method draws an `r_k`-dimensional random subspace in the
orthogonal complement of each estimated rank-`r_k` signal space. Consequently,
for automatic rank estimation this implementation requires both `p_k ≥ 2r_k`
and `n ≥ 2r_k` for every block. A supplied `joint_rank` bypasses Wedin
resampling and therefore does not impose this additional condition.

"""
function ajive(Xs::AbstractVector{<:AbstractMatrix{<:Real}};
        init_ranks::Vector{Int},
        joint_rank::Union{Nothing,Int,Symbol} = :auto,
        indiv_ranks::Union{Nothing,Vector{Int}} = nothing,
        orientation::Symbol = :features_by_samples,
        center::Bool = true,
        check_joint_identifiability::Bool = true,
        n_rand_samples::Int = 1000,
        rand_percentile::Real = 0.95,
        n_wedin_samples::Int = 1000,
        wedin_percentile::Real = 0.05,
        rng::AbstractRNG = Random.default_rng())

    # Validate the input parameters for the AJIVE algorithm
    joint_rank_fixed = if joint_rank === nothing || joint_rank === :auto
        nothing
    elseif joint_rank isa Int
        joint_rank
    else
        throw(ArgumentError("joint_rank must be an integer, nothing, or :auto"))
    end

    # Prepare the data blocks: center if requested, ensure Float64 type, 
    # and check dimensions
    Xc, means, n = _ajive_prepare_blocks(Xs; orientation = orientation, center = center)
    _ajive_check_ranks(Xc, init_ranks, joint_rank_fixed, indiv_ranks)

    # Step 1: initial signal-space extraction.
    Us, svals, Vs, thresholds, init_svals =
        _ajive_initial_signal_svd(Xc, init_ranks)

    # Step 2: flag-mean/common score-space SVD.
    Vcommon, common_svals = _ajive_common_score_space(Vs, init_ranks, n)

    # Step 2a: estimate the joint rank if not fixed by the user. 
    # The candidate joint rank is the number of common-space singular 
    # values above the larger of the random-direction and Wedin bounds.
    rank_diagnostics = nothing
    candidate_r = if joint_rank_fixed === nothing
        rhat, diagnostics = _ajive_estimate_joint_rank(
            Xc, Us, svals, Vs, common_svals, n, init_ranks;
            n_rand_samples = n_rand_samples,
            rand_percentile = rand_percentile,
            n_wedin_samples = n_wedin_samples,
            wedin_percentile = wedin_percentile,
            rng = rng)
        rank_diagnostics = diagnostics
        rhat
    else
        joint_rank_fixed
    end

    # Step 2b: compute the joint space.
    VJ = candidate_r == 0 ? zeros(Float64, n, 0) :
        Matrix(@view Vcommon[:, 1:candidate_r])

    # Step 3a: ensure every candidate direction is signal in every block.
    dropped_joint = Int[]
    if check_joint_identifiability && candidate_r > 0
        VJ, dropped_joint = _ajive_check_identifiability(Xc, VJ, thresholds)
    end
    r = size(VJ, 2)

    # Step 3b: block-specific joint, individual, and residual matrices.
    J, Iblocks, E, U, Si, Wi, ri = _ajive_final_decomposition(
        Xc, VJ, thresholds; indiv_ranks = indiv_ranks)

    # Compute the common normalized scores matrix S as the transpose of VJ    
    S = Matrix{Float64}(transpose(VJ))

    return Ajive{Float64}(J, Iblocks, E, S, U, Si, Wi,
        copy(init_ranks), candidate_r, r, ri, thresholds, init_svals,
        common_svals, dropped_joint, rank_diagnostics, means, center, orientation)
end

"""
    ajive(Xs, init_ranks, joint_rank; kwargs...)

Positional convenience form for supplying the initial signal ranks and joint
rank. Equivalent to `ajive(Xs; init_ranks=init_ranks, joint_rank=joint_rank,
kwargs...)`.
"""
ajive(Xs::AbstractVector{<:AbstractMatrix{<:Real}},
    init_ranks::Vector{Int}, joint_rank::Int; kwargs...) =
    ajive(Xs; init_ranks = init_ranks, joint_rank = joint_rank, kwargs...)