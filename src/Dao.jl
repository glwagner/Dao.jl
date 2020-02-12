module Dao

export # markov.jl
    NegativeLogLikelihood,
    BatchedNegativeLogLikelihood,
    NLL,
    MarkovLink,
    MarkovChain,
    extend!,
    errors,
    params,
    optimal,
    status,
    paramtype,
    paramnames,
    paramindex,

    # samplers.jl
    MetropolisSampler,
    NormalPerturbation,
    BoundedNormalPerturbation,

    # utils.jl
    collect_samples,
    anneal,
    AdaptiveAlgebraicSchedule,
    AdaptiveExponentialSchedule,
    estimate_covariance,
    initialize_bounds,
    estimate_bounds

using
    Printf,
    Random,
    Distributions,
    Statistics,
    JLD2

import Base: length, getindex, lastindex

import Statistics: cov

abstract type AbstractNegativeLogLikelihood <: Function end
const ANLL = AbstractNegativeLogLikelihood

"""
    NegativeLogLikelihood(model, data, loss; kwargs...)

Construct a function that compute the negative log likelihood
of the parameters `x` given `model, `data`, and a prior
parameter distribution `prior`.

The `loss` function has the calling signature

    `loss(θ, model, data)`,

when `weights` are `nothing`, or

    `loss(θ, model, data, weights)`,

where `θ` is a parameters object.

The keyword arguments permit the user to specify

* `scale`
* `prior`
* `weights`
* `output`

"""
mutable struct NegativeLogLikelihood{P, M, D, L, T} <: ANLL
      model :: M
       data :: D
       loss :: L
      scale :: T
      prior :: P
end

NegativeLogLikelihood(model, data, loss; scale=1.0, prior=nothing) =
    NegativeLogLikelihood(model, data, loss, scale, prior)

const NLL = NegativeLogLikelihood

# NLL signature with no prior
(nll::NLL{<:Nothing})(θ) = nll.loss(θ, nll.model, nll.data)

mutable struct BatchedNegativeLogLikelihood{B, W, T} <: ANLL
      batch :: B
    weights :: W
      scale :: T
end

function BatchedNegativeLogLikelihood(batch; weights=[1.0 for b in batch], scale=1.0)
    return BatchedNegativeLogLikelihood(batch, weights, scale)
end

const BNLL = BatchedNegativeLogLikelihood

function (bl::BNLL)(θ)
    @inbounds begin
        total_err = bl.weights[1] * bl.batch[1](θ)
        for i = 2:length(bl.batch)
            total_err += bl.weights[i] * bl.batch[i](θ)
        end
    end
    return total_err
end

include("samplers.jl")
include("markov.jl")
include("utils.jl")
include("annealing.jl")

end # module
