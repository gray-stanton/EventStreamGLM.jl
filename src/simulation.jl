struct PoissonProcess{T <: Real}
    intensity :: Union{T, Function}
    maxtime :: T
end

struct HomogPoissonProcess{T <: Real}
    intensity :: T
    maxtime :: T
end

struct HawkesProcess{T <: Real}
    α :: T
    β :: T
    μ :: T
    maxtime
end


struct EventStreamProcess{T <: Real, L}
    other_events :: Vector{Tuple{T, L}}
    labels :: Vector{L}
    maxtime :: T
    basis :: BSplineBasis
    memory ::T
    λ_0 ::T
    other_kernels :: Array{Spline{BSplineBasis{Vector{T}}, Vector{T}}, 1}
    self_kernel :: Spline
end

function other_intensity(p :: EventStreamProcess, t)
    label_order = Dict(l => i for (i, l) in enumerate(p.labels))
    could_influence = searchsorted(p.other_events, prevfloat(t), by=(tup) -> memorylengths_away(tup[1], t, p.memory))
    influences = [(t - e[1], label_order[e[2]]) for e in p.other_events[could_influence]]
    total_intensity = sum([p.other_kernels[infl[2]](infl[1]) for infl in influences])
    return total_intensity
end


function rand(p :: EventStreamProcess, intensity_ub::Real)
    # Hopefully can remove specification of intensity_ub, as difficult to specify
    # TODO: switch to Orgata, use fact that upper-bound for each spline is max(coefs)
    points = eltype(p.maxtime)[]
    s = 0
    while s < p.maxtime
        u, d = rand(Float64, 2) # Uniform(0, 1)
        w = -log(u)/intensity_ub # Exponential with mean 1/intensity_ub
        s = s + w # Proposed point
        s <=p.maxtime || break
        influential_diffs = eltype(p.maxtime)[]
        for x in Iterators.reverse(points)
            s - x > p.memory || break
            push!(influential_diffs, s - x)
        end
        int_value = other_intensity(p, s) + sum(p.self_kernel.(influential_diffs))
        int_value = p.λ_0 * exp.(int_value)
        if int_value > intensity_ub
            @warn "Intensity Upper Bound incorrectly specified: λ($s) = $int_value > $intensity_ub"
        end
        if d <= int_value/intensity_ub
            # accept with prop int_value/intensity_ub
            push!(points, s)
        end
        #ow reject
    end
    return points
end

function rand(p :: HawkesProcess)
    # Ogata Iterative Thinning
    points = eltype(p.α)[]
    s = 0
    while s < p.maxtime
        local_ub = p.μ + sum(p.α .* exp.(-1*p.β*(s .- points))) # Local UB on conditional intensity function, assuming no more points
        u, d = rand(Float64, 2)
        w = -log(u)/local_ub
        s= s + w
        s <=p.maxtime || break
        int_value = p.μ + sum(p.α .* exp.(-1*p.β*(s .- points)))
        if int_value > local_ub
            @warn "Intensity Upper Bound incorrectly specified: λ($s) = $int_value > $local_ub"
        end
        if d <= int_value/local_ub
            push!(points, s)
        end
    end
    return points
end

function rand(p :: HomogPoissonProcess)
    points = eltype(p.intensity)[]
    s=0
    while s<p.maxtime
        u = rand(Float64)
        w = -log(u)/p.intensity # Exponential wiht mean 1/intensity
        s = s+w
        s <= p.maxtime || break
        push!(points, s)
    end
    return points
end


function rand(p :: PoissonProcess, intensity_ub )
    # Shedler-Lewis Thinning
    points = eltype(intensity_ub)[]
    s = 0
    while s<p.maxtime
        u, d = rand(Float64, 2) # Uniform(0, 1)
        w = -log(u)/intensity_ub # Exponential with mean 1/intensity_ub
        s = s + w # Proposed point
        s <=p.maxtime || break
        int_value = p.intensity(s) :: Float64
        if int_value > intensity_ub
            @warn "Intensity Upper Bound incorrectly specified: λ($s) = $int_value > $intensity_ub"
        end
        if d <= int_value/intensity_ub
            # accept with prop int_value/intensity_ub
            push!(points, s)
        end
        #ow reject
    end
    return points
end

