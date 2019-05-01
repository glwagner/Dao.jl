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
mutable struct NegativeLogLikelihood{P, W, M, D, T}
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

mutable struct BatchedNegativeLogLikelihood{P, W, M, D, T, BW}
      batch :: Vector{NLL{P, W, M, D, T}}
    weights :: BW
end

BatchedNegativeLogLikelihood(batch) = BatchedNegativeLogLikelihood(batch, (1.0 for b in batch))

const BNLL = BatchedNegativeLogLikelihood

function (bl::BNLL)(𝒳)
    @inbounds begin
        total_err = bl.weights[1] * bl.batch[1].loss(𝒳)
        for i = 2:length(bl.batch)
            total_err += bl.weights[i] * bl.batch[i].loss(𝒳)
        end
    end

    return total_err
end

struct MarkovLink{T, P}
    param :: P
    error :: T
end

MarkovLink(T, nll, param) = MarkovLink{T, typeof(param)}(param, nll(param))
MarkovLink(nll::NLL, param) = MarkovLink(Float64, nll, param)

"""
    MarkovChain(nlinks, first_link, error_scale, nll, perturb)

Generate a `MarkovChain` with `nlinks`, starting from `first_link`,
using the `nll` function to compute errors with `error_scale`,
and generating new parameters with `perturb`.
"""
mutable struct MarkovChain{T, X, L, P}
         links :: Vector{MarkovLink{T, X}}
          path :: Vector{Int}
           nll :: L
       sampler :: P
    acceptance :: Float64
end

import Base: getindex, length

getindex(chain::MarkovChain, inds...) = getindex(links, inds...)
length(chain::MarkovChain) = length(chain.path)

errors(chain::MarkovChain; after=1) = map(x -> x.error, view(chain.links, after:length(chain)))

function params(chain::MarkovChain{T, X}; after=1, matrix=false) where {T, X}
    paramvector = map(x -> x.param, view(chain.links, after:length(chain)))
    return matrix ? reinterpret(eltype(X), paramvector) : paramvector
end

function MarkovChain(nlinks::Int, first_link, nll, sampler)
    links = [first_link]
    path = Int[]
    markov_chain = MarkovChain(links, path, nll, sampler, 0.0)
    extend!(markov_chain, nlinks-1)
    return markov_chain
end

function extend!(nlinks, links, path, current, nll, sampler::MetropolisSampler)
    accepted = 0
    for i = 1:nlinks
        proposal = MarkovLink(nll, sampler.perturb(current.param))
        current = ifelse(accept(proposal, current, nll.scale), proposal, current)
        push!(links, proposal)
        if current === proposal
            accepted += 1
            push!(path, i)
        else
            @inbounds push!(path, path[end])
        end
    end

    return accepted
end

function extend!(chain, nlinks)
    accepted₀ = length(chain) * chain.acceptance
    accepted₊ = extend!(nlinks, chain.links, chain.path,
                                     chain.links[end], chain.nll, chain.sampler)
    chain.acceptance = (accepted₀ + accepted₊) / length(chain)
    return nothing
end

accept(new, current, scale) = current.error - new.error < scale * log(rand(Uniform(0, 1)))

function optimal(chain)
    iopt = argmin(errors(chain))
    return chain.links[iopt].param, chain.links[iopt].error
end
