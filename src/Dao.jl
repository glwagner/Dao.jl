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
    NonNegativeNormalPerturbation,

    # column_models/
    ColumnModels

using
    Printf,
    Random,
    Distributions,
    Statistics,
    JLD2

import Base: length

abstract type AbstractNegativeLogLikelihood <: Function end
const ANLL = AbstractNegativeLogLikelihood

"""
    NegativeLogLikelihood(model, data, loss; kwargs...)

Construct a function that compute the negative log likelihood
of the parameters `x` given `model, `data`, and a prior
parameter distribution `prior`.

The `loss` function has the calling signature

    `loss(x, model, data)`,

when `weights` are `nothing`, or

    `loss(x, model, data, weights)`,

where `x` is a parameters object.

The keyword arguments permit the user to specify

* `scale`
* `prior`
* `weights`

"""
mutable struct NegativeLogLikelihood{P, W, M, D, T} <: ANLL
      model :: M
       data :: D
       loss :: Function
      scale :: T
      prior :: P
    weights :: W
end

function NegativeLogLikelihood(model, data, loss;
                               scale=1.0, prior=nothing, weights=nothing)
    return NegativeLogLikelihood(model, data, loss, scale, prior, weights)
end

const NLL = NegativeLogLikelihood

(l::NLL{<:Nothing, <:Nothing})(𝒳) = l.loss(𝒳, l.model, l.data)
(l::NLL{<:Nothing})(𝒳) = l.loss(𝒳, l.model, l.data, l.weights)

mutable struct BatchedNegativeLogLikelihood{P, W, M, D, T, BW} <: ANLL
      batch :: Vector{NLL{P, W, M, D, T}}
    weights :: BW
      scale :: T
end

function BatchedNegativeLogLikelihood(batch; weights=[1 for b in batch], scale=1.0)
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

# Please don't look in this file.
include("old_markov.jl")

end # module
