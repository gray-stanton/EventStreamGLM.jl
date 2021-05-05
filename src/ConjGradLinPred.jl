
struct CGControl
    abstol :: Real
    reltol :: Real
    maxiter :: Int
    log :: Bool
    verbose :: Bool
    Pl
end


mutable struct EventStreamPredConjGrad{T, L} <: LinPred
    X :: AbstractEventStreamMatrix{T, L}
    beta0 :: Vector{T}
    delbeta :: Vector{T}
    scratchbeta :: Vector{T}
    control :: CGControl
    function EventStreamPredConjGrad{T, L}(E::AbstractEventStreamMatrix{T, L}, beta0::Vector{T}, cntrl::CGControl) where {T, L}
        n, p = size(E)
        length(beta0) == p || throw(DimensionMismatch("length(β0) ≠ size(X,2)"))
        new{T, L}(E, beta0, zeros(eltype(E.δ),p), zeros(eltype(E.δ),p), cntrl)
    end
    function EventStreamPredConjGrad{T, L}(E::AbstractEventStreamMatrix{T, L}, beta0::Vector{T}) where {T, L}
        n, p = size(E)
        length(beta0) == p || throw(DimensionMismatch("length(β0) ≠ size(X,2)"))
        cntrl = CGControl(zero(real(eltype(E.δ))), sqrt(eps(real(eltype(E.δ)))), size(E, 2), false, false, Identity())
        new{T, L}(E.δ, beta0, zeros(eltype(E.δ),p), zeros(eltype(E.\delta),p), cntrl)
    end
    function EventStreamPredConjGrad{T, L}(E::AbstractEventStreamMatrix{T, L}) where {T, L}
        n, p = size(E)
        cntrl = CGControl(zero(real(eltype(E))), sqrt(eps(real(eltype(E)))), size(E, 2), false, false, Identity())
        new{T, L}(E, zeros(eltype(E.δ), p), zeros(eltype(E.δ),p), zeros(eltype(E.δ),p), cntrl)
    end
end

EventStreamPredConjGrad(E :: AbstractEventStreamMatrix) = EventStreamPredConjGrad{eltype(fineness(E)), eltype(E.labels)}(E)


mutable struct DensePredConjGrad{T<:BlasReal} <: DensePred
    X :: Matrix{T}
    beta0 :: Vector{T}
    delbeta :: Vector{T}
    scratchbeta :: Vector{T}
    control :: CGControl
    intercept :: Bool
    function DensePredConjGrad{T}(X::Matrix{T}, beta0::Vector{T}, cntrl::CGControl) where T
        n, p = size(X)
        length(beta0) == p || throw(DimensionMismatch("length(β0) ≠ size(X,2)"))
        new{T}(X, beta0, zeros(eltype(X),p), zeros(eltype(X),p), cntrl, true)
    end
    function DensePredConjGrad{T}(X::Matrix{T}, beta0::Vector{T}) where T
        n, p = size(X)
        length(beta0) == p || throw(DimensionMismatch("length(β0) ≠ size(X,2)"))
        cntrl = CGControl(zero(real(eltype(X))), sqrt(eps(real(eltype(X)))), size(X, 2), false, false, Identity())
        new{T}(X, beta0, zeros(eltype(X),p), zeros(eltype(X),p), cntrl, true)
    end
    function DensePredConjGrad{T}(X::Matrix{T}) where T
        n, p = size(X)
        cntrl = CGControl(zero(real(eltype(X))), sqrt(eps(real(eltype(X)))), size(X, 2), false, false, Identity())
        new{T}(X, zeros(eltype(X), p), zeros(eltype(X),p), zeros(eltype(X),p), cntrl, true)
    end
end

DensePredConjGrad(X::Matrix) = DensePredConjGrad{eltype(X)}(X)


#TODO: Make these efficient
cholesky!(p::DensePredConjGrad) = cholesky(p.X'*p.X) 
cholesky(p::DensePredConjGrad) = cholesky(p.X'*p.X) 

cholesky(p :: EventStreamPredConjGrad) = Identity() #cholesky(Hermitian(XtWX(p.X, ones(p.X.nbins))))
cholesky!(p :: EventStreamPredConjGrad) = Identity() #cholesky(Hermitian(XtWX(p.X, ones(p.X.nbins))))

# Standard errors are way off
invchol(p :: EventStreamPredConjGrad) = inv(cholesky(p))


function linpred!(out, p :: EventStreamPredConjGrad, f::Real=1.0)
    out[:] = XWb(p.X, ones(size(p.X)[2]), p.beta0 + p.delbeta)
end



function delbeta!(p :: DensePredConjGrad{T}, r::Vector{T}) where T<: BlasReal
    delbeta!(p, r, ones(eltype(r), length(r)))
end

function delbeta!(p :: EventStreamPredConjGrad{T, L}, r :: Vector{T}, wt :: Vector{T}) where {T, L}
    G = WeightedGramMatrix(p.X, wt)
    p.delbeta = cg(G, XtWy(p.X, wt, r),
        abstol=p.control.abstol,
        reltol=p.control.reltol,
        maxiter=p.control.maxiter,
        log=p.control.log,
        verbose=p.control.verbose,
        Pl=p.control.Pl
    )
end 
 
function delbeta!(p :: DensePredConjGrad{T}, r :: Vector{T}, wt::Vector{T}) where T <: BlasReal
    G = WeightedNormGramMatrix(p.X, wt)
    p.delbeta = cg(G, G.X' * Diagonal(wt) * r ; 
        abstol=p.control.abstol,
        reltol=p.control.reltol,
        maxiter=p.control.maxiter,
        log=p.control.log,
        verbose=p.control.verbose,
        Pl=p.control.Pl
        )
    return p
end