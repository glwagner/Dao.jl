"""
    MarkovLink(nll, param)

Calculate a link in the Markov chain, using the negative log likelihood
function `nll` to calculate the negative log likelihood or 'error'
associated with `param`.
"""
struct MarkovLink{T, X}
    param :: X
    error :: T
    function MarkovLink(nll::Function, param)
        new{Float64, typeof(param)}(param, nll(param))
    end
end

paramtype(::MarkovLink{T, X}) where {T, X} = X
paramnames(::MarkovLink{T, X}) where {T, X} = fieldnames(X)

"""
    MarkovChain(nlinks, first_link, error_scale, nll, perturb)

Generate a `MarkovChain` with `nlinks`, starting from `first_link`,
using the `nll` function to compute errors with `error_scale`,
and generating new parameters with `perturb`.
"""
mutable struct MarkovChain{T, X, L, S}
         links :: Vector{MarkovLink{T, X}}
          path :: Vector{Int}
           nll :: L
       sampler :: S
    acceptance :: Float64
end

getindex(chain::MarkovChain, inds...) = getindex(chain.links, inds...)
length(chain::MarkovChain) = length(chain.path)
lastindex(chain::MarkovChain) = length(chain)

errors(chain::MarkovChain; after=1) = map(x -> x.error, view(chain.links, after:length(chain)))

function params(chain::MarkovChain{T, X}; after=1, matrix=false) where {T, X}
    paramvector = map(x -> x.param, view(chain.links, after:length(chain)))
    return matrix ? reinterpret(eltype(X), paramvector) : paramvector
end

paramtype(::MarkovChain{T, X}) where {T, X} = X
paramnames(::MarkovChain{T, X}) where {T, X} = fieldnames(X)
paramindex(p, chain) = findlast(θ->θ==p, paramnames(chain))

function vectorize(chain; after=1, finite_errors=false)
    errs = errors(chain, after=after)
    prms = params(chain, after=after)
    if finite_errors
        valids = isfinite.(errs)
        return errs[valids], prms[valids]
    else
        return errs, prms
    end
end

function MarkovChain(nlinks::Int, first_link, nll, sampler)
    links = [first_link]
    path = Int[0]
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

accept(new, current, scale) = current.error - new.error > scale * log(rand(Uniform(0, 1)))

optimal(chain) = chain[argmin(errors(chain))]

function status(chain::MarkovChain)
    return @sprintf("""
                   length | %d
               acceptance | %.9f
     initial scaled error | %.9f
     optimal scaled error | %.9f
    """, length(chain), chain.acceptance, chain[1].error/chain.nll.scale,
            optimal(chain).error/chain.nll.scale)
end
