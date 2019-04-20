struct MarkovLink{T, P}
    param :: P
    error :: T
end

MarkovLink(loss::Function, param) = MarkovLink(param, loss(param))
MarkovLink(n::Int, T=Float64) = MarkovLink(zeros(T, n), zero(T))

function like_direction(new_link, current_link, scale)
    return log(rand(Uniform(0, 1))) < (new_link.error - current_link.error) / scale
end

struct LossFunction{M, D, T} <: Function
           model :: M
            data :: D
    compute_loss :: Function
           scale :: T
end

(l::LossFunction)(params) = l.compute_loss(params, l.model, l.data)

"""
    MarkovChain(nlinks, first_link, error_scale, loss, perturb)

Generate a `MarkovChain` with `nlinks`, starting from `first_link`,
using the `loss` function to compute errors with `error_scale`,
and generating new parameters with `perturb`.
"""
struct MarkovChain{T, P, M, D}
       links :: Vector{MarkovLink{T, P}}
        path :: Vector{Int}
        loss :: LossFunction{M, D}
     perturb :: Function
end

function MarkovChain(nlinks, first_link, loss, perturb)
    links = [first_link]
    path = [0]

    current_link = first_link

    for i = 1:nlinks
        new_link = MarkovLink(loss, perturb(current_link.param))

        if like_direction(new_link, current_link, loss.scale)
            current_link = new_link
            push!(path, i) # next link originates from here
        else
            push!(path, path[end]) # next link originates in same place as previous
        end

        push!(links, new_link)
    end

    return MarkovChain(links, path, loss, perturb)
end

#=
struct MarkovWalker{M, D, T}
    loss :: LossFunction{M, D}
    perturb :: Function
    error_scale :: T
end

function MarkovChain(nlinks, first_link, walker)
    return MarkovChain(nlinks, first_link, walker.loss, walker.perturb;
                       error_scale=walker.error_scale)
end
=#
