"""
    NegativeLogLikelihood(model, data, compute_nll, scale=1)

Construct a function that compute the negative log likelihood
of the parameters `x` of `model`, given `data`.

The function `compute_nll` must have calling signature

`compute_nll(x, model, data)`,

where `x` is a parameters object.
"""
mutable struct NegativeLogLikelihood{P, M, D, T}
          model :: M
           data :: D
    compute_nll :: Function
          scale :: T
          prior :: P
end

# Add constructors with a default `scale` and `prior`.
function NegativeLogLikelihood(model, data, compute_nll, scale=1, prior=nothing)
    return NegativeLogLikelihood(model, data, compute_nll, 1, prior)
end

const NLL = NegativeLogLikelihood

(l::NLL{<:Nothing})(𝒳) = l.compute_nll(𝒳, l.model, l.data)

struct MarkovLink{T, P}
    param :: P
    error :: T
end

MarkovLink(T, nll, param) = MarkovLink{T, typeof(param)}(param, nll(param))
MarkovLink(nll, param) = MarkovLink(Float64, nll, param)

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

function errors(chain::MarkovChain)
    return map(x -> x.error, chain.links)
end

function params(chain::MarkovChain)
    return map(x -> x.param, chain.links)
end

function MarkovChain(nlinks::Int, first_link, nll, sampler)
    links = [first_link]
    path = Int[]
    markov_chain = MarkovChain(links, path, nll, sampler, 0.0)
    extend_markov_chain!(markov_chain, nlinks-1)
    return markov_chain
end

function extend_markov_chain!(nlinks, links, path, current, nll, sampler::MetropolisSampler)
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

function extend_markov_chain!(chain, nlinks)
    accepted₀ = length(chain) * chain.acceptance
    accepted₊ = extend_markov_chain!(chain, nlinks)
    chain.acceptance = (accepted₀ + accepted₊) / length(chain)
    return extend_markov_chain!(nlinks, chain.links, chain.path,
                                chain.links[end], chain.nll, chain.sampler)
end

accept(new, current, scale) = current.error - new.error > scale * log(rand(Uniform(0, 1)))
