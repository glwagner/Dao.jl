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
        return new{Float64, typeof(param)}(param, nll(param))
    end
end

paramtype(::MarkovLink{T, X}) where {T, X} = X
paramnames(::MarkovLink{T, X}) where {T, X} = fieldnames(X)

function Base.show(io::IO, link::MarkovLink{T, X}) where {T, X}
     names = [@sprintf("%-8s", name) for name in paramnames(link)]
    values = [@sprintf("%-8.4f", getproperty(link.param, name)) for name in paramnames(link)]

    return print(io, 
                 @sprintf("%d-parameter MarkovLink{%s, %s}:\n", length(link.param), T, X),
                 @sprintf("% 18s: %.6e\n", "error", link.error),
                 @sprintf("% 18s: ", "parameter names"), names..., '\n',
                 @sprintf("% 18s: ", "parameter values"), values...)
end

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
cov(chain::MarkovChain) = cov(collect_samples(chain), dims=2)

errors(chain::MarkovChain; after=1) = map(x -> x.error, view(chain.links, after:length(chain)))

Base.show(io::IO, chain::MarkovChain{T, X}) where {T, X} =
    print(io,
          @sprintf("%d-sample, %d-parameter MarkovChain{%s, %s}:\n", 
                   length(chain), length(chain[1].param), T, X),
          @sprintf("              acceptance: %.3f\n", chain.acceptance),
          @sprintf("    initial scaled error: %.3f\n", chain[1].error / chain.nll.scale),
          @sprintf("    optimal scaled error: %.3f\n", optimal(chain).error / chain.nll.scale))

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

"""
    extend!(chain, nlinks)
    
Extend a Markov `chain` by `nlinks`.
"""
function extend!(chain, nlinks)
    accepted₀ = length(chain) * chain.acceptance
    accepted₊ = extend!(nlinks, chain.links, chain.path, chain.links[end], chain.nll, chain.sampler)
    chain.acceptance = (accepted₀ + accepted₊) / length(chain)
    return nothing
end

accept(proposal, current, scale) = current.error - proposal.error > scale * log(rand(Uniform(0, 1)))

optimal(chain) = chain[argmin(errors(chain))]

function status(chain, iter=nothing, wall_time=nothing)

    covariance = cov(chain)
    variances = [covariance[i, i] for i = 1:size(covariance)[1]]
    
    iter != nothing && @printf("% 24s | %d   \n", "iter", iter)
    wall_time != nothing && @printf("% 24s | %s   \n", "wall time", prettytime(wall_time))

    @printf("% 24s | %d   \n", "samples", length(chain))
    @printf("% 24s | %.3f \n", "acceptance", chain.acceptance)
    @printf("% 24s | %.3e \n", "temperature", chain.nll.scale)
    @printf("% 24s | %.6f \n", "scaled optimal error", optimal(chain).error / chain[1].error)
    @printf("% 24s | %.6e \n", "unscaled optimal error",optimal(chain).error)
    @printf("% 24s | ", "parameter names"); [@printf("%-8s", n) for n in paramnames(chain[1])]
    @printf("\n")
    @printf("% 24s | ", "variances"); [@printf("%-8.4f", v) for v in variances]
    @printf("\n")
    @printf("% 24s | ", "optimal values"); [@printf("%-8.4f", p) for p in optimal(chain).param]
    @printf("\n\n")
    
    return nothing
end
