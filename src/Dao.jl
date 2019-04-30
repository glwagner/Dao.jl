module Dao

export # this file:
    markov_link,
    markov_chain,

    # markov.jl
    MarkovLink,
    MarkovChain,
    NegativeLogLikelihood,
    NLL,
    errors,
    params,

    # perturbations.jl
    NormalPerturbation,

    # column_models/
    ColumnModels

using
    Distributions,
    Statistics,
    Random,
    JLD2

import Base: length

include("perturbations.jl")
include("markov.jl")

include("old_markov.jl")

end # module
