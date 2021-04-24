using EventStreamGLM
using Test
using LinearAlgebra
using IterativeSolvers
using GLM
using BSplines
using StatsFuns

@testset "EventStreamGLM.jl" begin
    X1 = randn(1000, 10)
    lp1 = 1.0*X[:, 1] + 2.0 * X[:, 2]
    y_cts = lp1 + randn(1000)./sqrt(2)
    u = rand(length(lp1))
    y_bin = u .<= logistic.(lp1)
    pp0 = GLM.DensePredChol(X1, false) # Base GLM fitting
    rr0 = GLMResp(y_bin, Binomial(), LogitLink(), repeat([0.0], length(y_bin)), repeat([1.0], length(y_bin)))
    mod0 = GeneralizedLinearModel(pp0, rr0, false)
    fit!(mod0)
    evnts = rand(Float64, 150)
    labs = repeat(["a", "b"])
    eventstream = sort(collect(zip(evnts, labs)))
    X2 = FirstOrderBSplineEventStreamMatrix(eventstream, ["a", "b"], 0.001, 1.0, 4, [0.0, 0.1, 0.2])
    Q  = Matrix(X2)
    lp2 = 1.0*X2[:, 1] + 2.0*[:, 2]
    y_cts2 = lp2 + randn(1000)./sqrt(2)
    u2 = rand(length(lp2))
    y_bin2 = u.<logitistic.(lp2)
    @testset DensePredConjGrad begin
        pp1 = DensePredConjGrad(X1)
        rr1 = GLMResp(y_cts, Normal(), IdentityLink(),
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
        pp3 = EventStreamPredConjGrad(E)
        rr3 = GLMResp(y_cts2, Normal(), LogitLink, 
            repeat([0.0], length(y_cts2)), repeat([1.0], length(y_cts2)))
        mod3 = GeneralizedLinearModel(rr3, pp3, false)
        fit!(mod3)
        
        pp4 = EventStreamPredConjGrad(E)
        rr4 = GLMResp(y_bin2)
        #TODO finish these tests
    end
    @testset simulation begin
        basis = BSplineBasis(4, [0.0, 10.0, 20.0, 50.0, 100.0, 200.0])
        other_events = 10000 .* rand(Float64, 1000)
        labels = repeat(["a", "b"], 500)
        eventstream = sort(collect(zip(other_events, labels)))
        other_coefs = [randn(length(basis)) for _ in 1:2]
        self_coefs = randn(length(basis))
        other_kernels = [Spline(basis, c) for c in other_coefs]
        self_kernel = Spline(basis, self_coefs)
        P1 = EventStreamProcess(eventstream, ["a", "b"], 10000.0, 
            basis, 200.0, other_kernels, self_kernel)
    end

end
