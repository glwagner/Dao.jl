"""
    NegativeLogLikelihood(model, data, compute_logdensity, scale=1)

Construct a function that compute the negative log likelihood
of the parameters `x` of `model`, given `data`.

The function `compute_nll` must have calling signature

`compute_nll(x, model, data)`,

where `x` is a parameters object.
"""
struct NegativeLogLikelihood{P, M, D, T}
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

MarkovLink(T, logdensity::NegativeLogLikelihood, param) = MarkovLink{T, typeof(param)}(param, logdensity(param))
MarkovLink(logdensity::NegativeLogLikelihood, param) = MarkovLink(Float64, logdensity, param)

"""
    MarkovChain(nlinks, first_link, error_scale, logdensity, perturb)

Generate a `MarkovChain` with `nlinks`, starting from `first_link`,
using the `logdensity` function to compute errors with `error_scale`,
and generating new parameters with `perturb`.
"""
struct MarkovChain{T, X, L, P}
         links :: Vector{MarkovLink{T, X}}
          path :: Vector{Int}
    logdensity :: L
       sampler :: P
end

import Base: getindex

getindex(chain::MarkovChain, inds...) = getindex(links, inds...)

function errors(chain::MarkovChain)
    return map(x -> x.error, chain.links)
end

function params(chain::MarkovChain)
    return map(x -> x.param, chain.links)
end

function MarkovChain(nlinks::Int, first_link, logdensity, sampler)
    links = [first_link]
    path = Int[]
    _markov_chain!(nlinks, links, path, first_link, logdensity, sampler)
    return MarkovChain(links, path, logdensity, sampler)
end

function _markov_chain!(nlinks, links, path, current, logdensity, sampler::MetropolisSampler)
    for i = 1:nlinks
        proposal = MarkovLink(logdensity, sampler.perturb(current.param))
        current = ifelse(accept(proposal, current, logdensity.scale), proposal, current)
        push!(links, proposal)
        @inbounds push!(path, ifelse(current===proposal, i, path[end]))
    end

    return nothing
end

accept(new, current, scale) = current.error - new.error > scale * log(rand(Uniform(0, 1)))
