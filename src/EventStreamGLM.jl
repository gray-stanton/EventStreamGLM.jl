module EventStreamGLM

import GLM


import Base: rand
import GLM: BlasReal, DensePred, delbeta!, cholesky!, cholesky, LinPred, invchol, linpred!
import EventStreamMatrix: AbstractEventStreamMatrix, AbstractEventStreamVector


export DensePredConjGrad
export delbeta!
export cholesky!, cholesky
export CGControl
export EventStreamPredConjGrad
export rand
export greet
export EventStreamProcess
export other_intensity
export HomogPoissonProcess
export PoissonProcess
export HawkesProcess

using EventStreamMatrix
using GLM
using IterativeSolvers
using LinearAlgebra
using StatsFuns
using BSplines

greet() = "Damns"

include("ConjGradLinPred.jl")
include("simulation.jl")

end
