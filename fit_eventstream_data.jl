using ArgParse
using EventStreamMatrix
using EventStreamGLM
using YAML
using GLM
using BSplines
import Dates: now, Millisecond

import Random: seed!
import IterativeSolvers: Identity
function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--infolder"
            help="Folder to read in simulated dataframes from"
            default="/tmp/"
        "--outfolder"
            help="Folder to write out fitted models"
            default="/tmp/"
        "--seed"
            help="RNG Seed"
            default=564
            arg_type=Int
        "--name"
            help="Name to read in and write"
            default="sim"
        "--include_lag"
            help="Whether or not to use response variable lagged by one bin as additional input"
            action=:store_true
        "--dense"
            help="Whether to use a dense design matrix representation"
            action=:store_true
        "--nonconj"
            help="Whether to default to a non-CG fit"
            action=:store_true
        "--fineness"
            help="Level of discretization to use"
            default=0.1
            arg_type=Float64
    end
    return parse_args(s)
end

function main()
    args = parse_commandline()
    println("Parsed args:")
    for (arg,val) in args
        println("  $arg  =>  $val")
    end
    seed!(args["seed"])
    BLAS.set_num_threads(1)
    for file in readdir(abspath(args["infolder"]))
        if startswith(file, args["name"])
            println("Reading $file")
            dat = YAML.load_file(joinpath(abspath(args["infolder"]), file))
            fittime_start = now()
            outpoints = dat["output"]
            δ = args["fineness"]
            nsources=convert(Int, dat["nsources"][1])
            maxtime = convert(Float64, dat["maxtime"][1])
            order = convert(Int, dat["order"][1])
            eventstream = Tuple{Float64, Int}[]
            labels = ["source_$i" for i in 1:nsources] 
            for (i, l) in enumerate(labels)
                eventstream = vcat(eventstream, [(t, i) for t in dat[l]]) :: Vector{Tuple{Float64, Int}}
            end
            if args["include_lag"]
                push!(labels, "response_lag")
                # Lag by 
                lag_resp = outpoints .+ δ 
                filter!(t -> t<=maxtime, lag_resp)
                eventstream = vcat(eventstream, [(t, nsources + 1) for t in lag_resp])
            end
            sort!(eventstream)
            
            # Construct  data
            basis = BSplineBasis(dat["order"], dat["breakpoints"])
            E = FirstOrderBSplineEventStreamMatrix(eventstream, labels, δ, maxtime, order, dat["breakpoints"], true)
            y_es = IdentityEventStreamVector(outpoints, δ, maxtime)
            y = Vector(y_es)
            y = y .>= 1 ## For binary response

            # Construct GLM
            cntrl = CGControl(0.001, 0.001, 40, false, true, Identity())
            if args["dense"] && !args["nonconj"]
                pp=DensePredConjGrad{Float64}(hcat(ones(Float64, size(E)[1]),Matrix(E)), zeros(Float64, size(E)[2]+1) , cntrl)
            elseif !args["dense"] && !args["nonconj"]
                pp = EventStreamPredConjGrad{Float64, String}(E, zeros(Float64, size(E)[2]), cntrl)
            else
                pp = GLM.DensePredChol(hcat(ones(Float64, size(E)[1]), Matrix(E)), false)
            end
            rr = GLM.GlmResp(y, Binomial(), LogitLink(), repeat([0.0], length(y)), repeat([1.0], length(y)))
            mod = GeneralizedLinearModel(rr, pp, false)

            # Fitting!
            print("Fitting $file")
            fit!(mod)
            fittime_end = now()
            fittime = Millisecond(fittime_end - fittime_start).value

            # Save output
            outdata = Dict{String, Vector{Float64}}("fineness" => [δ])
            outdata["coefs"] = coef(mod)
            outdata["fittime"] = fittime
            YAML.write_file(joinpath(abspath(args["outfolder"], file)), outdata)
        end
    end
end

main()