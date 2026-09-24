# Tests for the deterministic AJIVE core.  The automatic random-direction and
# Wedin joint-rank estimator is intentionally outside this first milestone.

function make_ajive_data(; seed = 321)
    Random.seed!(seed)
    n = 12
    p1, p2 = 9, 8
    r = 2
    r1, r2 = 2, 2

    # Build exactly orthogonal score spaces: two joint directions followed by
    # two individual directions for each block.  QR makes the intended row
    # spaces exact while leaving arbitrary feature loadings.
    Q = Matrix(qr(randn(n, r + r1 + r2)).Q)[:, 1:(r + r1 + r2)]
    S = transpose(Q[:, 1:r])
    S1 = transpose(Q[:, (r + 1):(r + r1)])
    S2 = transpose(Q[:, (r + r1 + 1):(r + r1 + r2)])

    U1 = randn(p1, r)
    U2 = randn(p2, r)
    W1 = randn(p1, r1)
    W2 = randn(p2, r2)

    J1 = U1 * S
    J2 = U2 * S
    I1 = W1 * S1
    I2 = W2 * S2
    X1 = J1 + I1
    X2 = J2 + I2

    return (; X1, X2, J1, J2, I1, I2, S, r, r1, r2)
end

@testset "AJIVE deterministic core recovers exact joint and individual subspaces" begin
    d = make_ajive_data()
    m = BigRiverEssence.ajive([d.X1, d.X2];
        init_ranks = [d.r + d.r1, d.r + d.r2],
        joint_rank = d.r,
        center = false)

    @test m isa BigRiverEssence.Ajive
    @test m.r == d.r
    @test m.ri == [d.r1, d.r2]

    # AJIVE bases can rotate within a repeated-singular-value joint subspace, so
    # compare projection matrices rather than individual score vectors.
    P_est = transpose(m.S) * m.S
    P_true = transpose(d.S) * d.S
    @test norm(P_est - P_true) < tol_ord

    @test norm(m.J[1] - d.J1) < 1e-7
    @test norm(m.J[2] - d.J2) < 1e-7
    @test norm(m.I[1] - d.I1) < 1e-7
    @test norm(m.I[2] - d.I2) < 1e-7

    # Exact reconstruction and joint ⟂ individual score spaces.
    for b in 1:2
        X = b == 1 ? d.X1 : d.X2
        @test norm(X - m.J[b] - m.I[b] - m.E[b]) < tol_ord
        @test norm(m.S * transpose(m.Si[b])) < 1e-7
        @test norm(m.J[b] - m.U[b] * m.S) < 1e-7
        @test norm(m.I[b] - m.Wi[b] * m.Si[b]) < 1e-7
    end
end

@testset "AJIVE orientation conversion gives the same canonical decomposition" begin
    d = make_ajive_data(seed = 99)

    a = BigRiverEssence.ajive([d.X1, d.X2];
        init_ranks = [4, 4], joint_rank = 2,
        orientation = :features_by_samples, center = true)

    b = BigRiverEssence.ajive([transpose(d.X1), transpose(d.X2)];
        init_ranks = [4, 4], joint_rank = 2,
        orientation = :samples_by_features, center = true)

    @test a.r == b.r
    @test a.ri == b.ri
    @test a.thresholds ≈ b.thresholds
    @test transpose(a.S) * a.S ≈ transpose(b.S) * b.S
    for k in 1:2
        @test a.J[k] ≈ b.J[k]
        @test a.I[k] ≈ b.I[k]
        @test a.E[k] ≈ b.E[k]
    end
end

@testset "AJIVE Step-1 thresholds use adjacent singular-value midpoints" begin
    d = make_ajive_data(seed = 7)
    Xc, _, _ = BigRiverEssence._ajive_prepare_blocks([d.X1, d.X2]; center = false)
    _, _, _, thresholds, init_svals =
        BigRiverEssence._ajive_initial_signal_svd(Xc, [4, 4])

    for b in 1:2
        @test thresholds[b] ≈ (init_svals[b][4] + init_svals[b][5]) / 2
    end
end

@testset "AJIVE joint rank zero" begin
    d = make_ajive_data(seed = 16)
    m = BigRiverEssence.ajive([d.X1, d.X2];
        init_ranks = [4, 4], joint_rank = 0, center = false)

    @test m.r == 0
    @test size(m.S) == (0, size(d.X1, 2))
    @test all(iszero, m.J[1])
    @test all(iszero, m.J[2])
    for b in 1:2
        X = b == 1 ? d.X1 : d.X2
        @test X ≈ m.I[b] + m.E[b]
    end
end

@testset "AJIVE argument validation" begin
    Random.seed!(12)
    X1 = randn(7, 10)
    X2 = randn(5, 10)

    @test_throws ArgumentError BigRiverEssence.ajive([X1]; init_ranks = [2], joint_rank = 1)
    @test_throws ArgumentError BigRiverEssence.ajive([X1, X2]; init_ranks = [2], joint_rank = 1)
    @test_throws ArgumentError BigRiverEssence.ajive([X1, X2]; init_ranks = [2, 2], joint_rank = 3)
    @test_throws ArgumentError BigRiverEssence.ajive([X1, randn(5, 9)]; init_ranks = [2, 2], joint_rank = 1)
    @test_throws ArgumentError BigRiverEssence.ajive([X1, X2]; init_ranks = [2, 2], joint_rank = 1, orientation = :bad)
end

@testset "AJIVE does not mutate caller inputs" begin
    d = make_ajive_data(seed = 44)
    X1 = copy(d.X1)
    X2 = copy(d.X2)
    X1_before = copy(X1)
    X2_before = copy(X2)

    BigRiverEssence.ajive([X1, X2]; init_ranks = [4, 4], joint_rank = 2)
    @test X1 == X1_before
    @test X2 == X2_before
end
