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
    BoundedNormalPerturbation

using
    Printf,
    Random,
    Distributions,
    Statistics,
    JLD2

import Base: length, getindex, lastindex

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
mutable struct NegativeLogLikelihood{P, W, O, M, D, T} <: ANLL
      model :: M
       data :: D
       loss :: Function
      scale :: T
      prior :: P
    weights :: W
     output :: O
end

function NegativeLogLikelihood(model, data, loss;
                               scale=1.0, prior=nothing, weights=nothing, output=nothing)
    return NegativeLogLikelihood(model, data, loss, scale, prior, weights, output)
end

const NLL = NegativeLogLikelihood

# Flavors of NLL signatures
(nll::NLL{<:Nothing, <:Nothing, <:Nothing})(θ) = nll.loss(θ, nll.model, nll.data)
(nll::NLL{<:Nothing, <:Nothing})(θ)            = nll.loss(θ, nll.model, nll.data, nll.output)
(nll::NLL{<:Nothing, W, <:Nothing})(θ) where W = nll.loss(θ, nll.model, nll.data, nll.weights)
(nll::NLL{<:Nothing})(θ)                       = nll.loss(θ, nll.model, nll.data, nll.weights, nll.output)

mutable struct BatchedNegativeLogLikelihood{B, W, T} <: ANLL
      batch :: B
    weights :: W
      scale :: T
end

function BatchedNegativeLogLikelihood(batch; weights=[1.0 for b in batch], scale=1.0)
    return BatchedNegativeLogLikelihood(batch, weights, scale)
end

const BNLL = BatchedNegativeLogLikelihood

function (bl::BNLL)(𝒳)
    @inbounds begin
        total_err = bl.weights[1] * bl.batch[1](𝒳)
        for i = 2:length(bl.batch)
            total_err += bl.weights[i] * bl.batch[i](𝒳)
        end
    end
    return total_err
end

include("samplers.jl")
include("markov.jl")

end # module
