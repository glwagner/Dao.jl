struct MarkovLink{T, P}
    param :: P
    error :: T
end

MarkovLink(T, loss::Function, param) = MarkovLink{T, typeof(param)}(param, loss(param))
MarkovLink(loss::Function, param) = MarkovLink(Float64, loss, param)


"""
    LossFunction(model, data, compute_loss, scale=1)

Construct a loss function that computes the error with `scale`
given `model`, `data`, and parameters `x`.

The function `compute_loss` must have calling signature

`compute_loss(x, model, data)`,

where `x` is a parameters object.
"""
struct LossFunction{M, D, T} <: Function
           model :: M
            data :: D
    compute_loss :: Function
           scale :: T
end

# Add a constructor whose default value for 'scale' is 1.
LossFunction(model, data, compute_loss) = LossFunction(model, data, compute_loss, 1)

(l::LossFunction)(x) = l.compute_loss(x, l.model, l.data)

"""
    MarkovChain(nlinks, first_link, error_scale, loss, perturb)

Generate a `MarkovChain` with `nlinks`, starting from `first_link`,
using the `loss` function to compute errors with `error_scale`,
and generating new parameters with `perturb`.
"""
struct MarkovChain{T, X, L, P}
       links :: Vector{MarkovLink{T, X}}
        path :: Vector{Int}
        loss :: L
     perturb :: P
end

import Base: getindex

getindex(chain::MarkovChain, inds...) = getindex(links, inds...)

function MarkovChain(nlinks::Int, first_link, loss, perturb)
    links = [first_link]
    path = Int[]
    _markov_chain!(nlinks, links, path, first_link, loss, perturb)
    return MarkovChain(links, path, loss, perturb)
end

function _markov_chain!(nlinks, links, path, current, loss, perturb)
    for i = 1:nlinks
        new = MarkovLink(loss, perturb(current.param))
        current = ifelse(accept(new, current, loss.scale), new, current)
        push!(links, new)
        @inbounds push!(path, ifelse(current===new, i, path[end]))
    end

    return nothing
end

accept(new, current, scale) = current.error - new.error > scale * log(rand(Uniform(0, 1)))

function errors(chain::MarkovChain)
    return map(x -> x.error, chain.links)
end

function params(chain::MarkovChain)
    return map(x -> x.param, chain.links)
end
