using EventStreamGLM
using EventStreamMatrix
using Test
using LinearAlgebra
using IterativeSolvers
using GLM
using BSplines
using StatsFuns

@testset "EventStreamGLM.jl" begin
    X1 = randn(1000, 10)
    lp1 = 1.0*X1[:, 1] + 2.0 * X1[:, 2]
    y_cts = lp1 + randn(1000)./sqrt(2)
    u = rand(length(lp1))
    y_bin = u .<= logistic.(lp1)
    pp0 = GLM.DensePredChol(X1, false) # Base GLM fitting
    rr0 = GLM.GlmResp(y_bin, Binomial(), LogitLink(), repeat([0.0], length(y_bin)), repeat([1.0], length(y_bin)))
    mod0 = GeneralizedLinearModel(rr0, pp0, false)
    fit!(mod0)
    evnts = rand(Float64, 150)
    labs = repeat([1, 2], 75)
    eventstream = sort(collect(zip(evnts, labs)))
    X2 = FirstOrderBSplineEventStreamMatrix(eventstream, ["a", "b"], 0.001, 1.0, 4, [0.0, 0.1, 0.2], true)
    Q  = Matrix(X2)
    lp2 = 1.0*Q[:, 1] + 2.0*Q[:, 2]
    y_cts2 = lp2 + randn(1000)./sqrt(2)
    u2 = rand(length(lp2))
    y_bin2 = u2.<logistic.(lp2)
    @testset DensePredConjGrad begin
        pp1 = DensePredConjGrad(X1)
        rr1 = GLM.GlmResp(y_cts, Normal(), IdentityLink(),
             repeat([0.0], length(y_cts)), repeat([1.0], length(y_cts)))
        mod1 = GeneralizedLinearModel(rr1, pp1, false)
        fit!(mod1)
        @test abs(coef(mod1)[1] - 1.0) <= 0.05
        @test abs(coef(mod1)[2] - 2.0) <= 0.05
        @test abs(coef(mod1)[3]) <= 0.05
        pp2 = DensePredConjGrad(X1)
        rr2 = GLMResp(y_bin, Binomial(), LogitLink(), repeat([0.0], length(y_bin)),
            repeat([1.0], length(y_bin)))
        mod2 = GeneralizedLinearModel(rr2, pp2, false)
        fit!(mod2)
        @test abs(coef(mod2)[1] - 1.0) <= 0.1
        @test abs(coef(mod2)[2] - 2.0) <= 0.1
        @test abs(coef(mod2)[3] - 0.0) <= 0.1
        @test sum(abs.(coef(mod2) - coef(mod0))) <= 0.0001
    end
    @testset EventStreamPredConjGrad begin
        pp3 = EventStreamPredConjGrad(X2)
        rr3 = GLM.GlmResp(y_cts2, Normal(), IdentityLink(), 
            repeat([0.0], length(y_cts2)), repeat([1.0], length(y_cts2)))
        mod3 = GeneralizedLinearModel(rr3, pp3, false)
        fit!(mod3)
        
        pp4 = EventStreamPredConjGrad(E)
        rr4 = GLMResp(y_bin2)
        #TODO finish these tests
    end
    @testset simulation begin
        basis = BSplineBasis(4, [0.0, 10.0, 20.0])
        other_events = 100000 .* rand(Float64, 10000)
        labels = repeat([1, 2], 5000)
        eventstream = sort(collect(zip(other_events, labels)))
        other_coefs = [randn(length(basis))/4 for _ in 1:2]
        self_coefs = randn(length(basis))
        other_kernels = [Spline(basis, c) for c in other_coefs]
        self_kernel = Spline(basis, min.(0, self_coefs))
        λ_0 = 0.4
        P1 = EventStreamProcess(eventstream, ["a", "b"], 100000.0, 
            basis, 20.0, λ_0, other_kernels, self_kernel)
    end

end
