using ArgParse
using EventStreamMatrix
using EventStreamGLM
using YAML
using BSplines

import Random: seed!
function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--outfolder"
            help="Folder to write out the simulated datasets"
            default="/tmp/"
        "--name"
            help="Simulated dataset file name prefix"
            default="sim"
        "--nsims"
            help="Number of simulated datasets to create"
            default=1
            arg_type=Int
        "--maxtime"
            help="Maximum amount of time to simulate for"
            default = 10.0
            arg_type=Float64
        "--intensity_ub"
            help="Upper bound for self-point intensity"
            default=10.0
            arg_type=Float64
        "--seed"
            help="RNG seed"
            default=534
            arg_type=Int
        "--config"
            help="Path to YAML configuration file"
            default="./config.yaml"
            arg_type=String
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
    conf = YAML.load_file(abspath(args["config"]))
    for (k, v) in pairs(conf)
        println("$k -> $v")
    end
    breakpoints = conf["breakpoints"]
    order = conf["order"]
    labels = ["source_$i" for i in 1:conf["inputnum"]] :: Vector{String}
    basis = BSplineBasis(order, breakpoints)
    # generate input events, one of ("homogpois")
    eventstream = Tuple{Float64, Int}[]
    if conf["input_type"] == "homogpois"
        for (j, intensity) in enumerate(conf["input_intensity"])
            P = HomogPoissonProcess(intensity, args["maxtime"])
            points = rand(P)
            evns = [(t, j) for t in points] :: Vector{Tuple{Float64, Int}}
            eventstream = vcat(eventstream, evns)
        end
    end
    sort!(eventstream) # ascending temporal order
    ## Create kernels
    other_kernels = [Spline(basis, q) for q in conf["other_coefs"]]
    self_kernel = Spline(basis, conf["self_coefs"])

    proc = EventStreamProcess(eventstream, labels, args["maxtime"],basis, support(basis)[2], conf["lambda_0"], other_kernels, self_kernel)
    # Sample 
    for j in 1:args["nsims"]
        fname = joinpath(args["outfolder"], "$(args["name"])_$(j).yaml")
        println("Simulating $fname")
        outpoints = rand(proc)
        out_data = Dict{String, Any}("nsources" => [length(labels)], "output" => outpoints)
        for (j,l) in enumerate(labels)
            out_data[l] = [e[1] for e in eventstream if e[2] == j]
        end
        # Locate some of the configuation data in each dataset.
        out_data["maxtime"] = [args["maxtime"]]
        out_data["true_other_coefs"] = conf["other_coefs"]
        out_data["true_self_coefs"] = conf["self_coefs"]
        out_data["breakpoints"] = conf["breakpoints"]
        out_data["order"] = conf["order"]
        YAML.write_file(fname, out_data)
    end
end


main()