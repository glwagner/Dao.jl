stagnant(chain) = chain[1].error == optimal(chain).error

get_samples(samples::Int, args...) = samples
get_samples(samples::Function, iter, args...) = samples(iter)

#
# Utilities for optimization
#
#

include("schedules.jl")

"""
    struct AnnealingProblem{N, M, P, R, S, I, A, C}

A problem in Simulated Annealing.
"""
struct AnnealingProblem{N, M, P, R, S, I, A, C}
    negative_log_likelihood :: N 
              markov_chains :: M
               perturbation :: P
          perturbation_args :: R
                    samples :: S
                 iterations :: I
         annealing_schedule :: A
        covariance_schedule :: C
end

function Base.show(io::IO, a::AnnealingProblem)
    print(io, "Dao.AnnealingProblem with markov_chains:\n\n")

    for chain in a.markov_chains
        show(io, chain)
        print(io, '\n')
    end

    print(io, "Optimal sample:\n\n")
    show(io, optimal(a.markov_chains[end]))

    return nothing
end

"""
    anneal(nll, initial_parameters, initial_covariance, 
           perturbation=NormalPerturbation, perturbation_args...; 
           samples=100, iterations=1, annealing_schedule=nothing, covariance_schedule=nothing)

Attempt to find the optimal parameter value that minimizes a negative log likelihood function 
`nll` using simulated-annealing-like procedure.
"""
function anneal(nll, initial_parameters, initial_covariance, 
                perturbation=NormalPerturbation, perturbation_args...; 
                samples=100, iterations=2, annealing_schedule=nothing, covariance_schedule=nothing)

    iterations > 1 || throw(ArgumentError("Annealing iterations must be > 1"))

    set_iterations!(annealing_schedule, iterations)
    set_iterations!(covariance_schedule, iterations)

    prob = AnnealingProblem(nll, [], perturbation, perturbation_args, samples, iterations, 
                            annealing_schedule, covariance_schedule)
                
    # Initialize optimization run
    iter = 1
    initial_link = MarkovLink(nll, initial_parameters)
    set_temperature!(annealing_schedule, initial_link)
    nsamples = get_samples(samples, iter, nothing)

    nll.scale, covariance = adjust_thermostat(annealing_schedule, covariance_schedule,
                                              initial_covariance, iter, initial_link, nothing)

    # Perform first mcmc run.
    chain = mcmc!(prob, iter, nsamples, initial_link, covariance)
    iter += 1

    # Annealing...
    while iter <= iterations
        println("Annealing (iteration: $iter)...\n")
        nsamples, initial_link, covariance = prepare_annealing(prob, iter, samples, covariance)
        chain = mcmc!(prob, iter, nsamples, initial_link, covariance)
        iter += 1
    end

    return prob
end

function mcmc!(p, iter, nsamples, initial_link, covariance)
    sampler = MetropolisSampler(p.perturbation(covariance, p.perturbation_args...))

    wall_time = @elapsed begin
        chain = MarkovChain(nsamples, initial_link, p.negative_log_likelihood, sampler)
    end
                                             
    status(chain, iter, wall_time)

    push!(p.markov_chains, chain)

    return chain
end

function prepare_annealing(prob, iter, samples, previous_covariance)

    # Choose next link based on optimal parameters
    chain = prob.markov_chains[end]
    parameters = optimal(chain).param
    link = MarkovLink(prob.negative_log_likelihood, parameters)

    # If acceptance is greater than 10%, use covariance estimate from prior chain
    # to choose next covariance. Otherwise, use the previous covariance estimate.
    covariance_estimate = cov(chain)

    # Use previous covariance if the estimated covariance matrix is not positive definite
    !isposdef(covariance_estimate) && (covariance_estimate = previous_covariance)

    temperature, covariance = adjust_thermostat(prob.annealing_schedule, prob.covariance_schedule,
                                                covariance_estimate, iter, link, chain)

    # Set the temperature for the next MCMC run
    prob.negative_log_likelihood.scale = temperature
    nsamples = get_samples(samples, iter, chain)

    return nsamples, link, covariance
end

function estimate_covariance(nll, initial_parameters, initial_covariance,
                             perturbation=NormalPerturbation, perturbation_args...; 
                             samples=100, schedule=nothing)

    initial_link = MarkovLink(nll, initial_parameters)
    nll.scale = initial_link.error
    covariance = initial_covariance

    sampler = MetropolisSampler(perturbation(covariance, perturbation_args...))
    chain = MarkovChain(samples, initial_link, nll, sampler)
    parameter_samples = collect_samples(chain)
    covariance = cov(parameter_samples, dims=2)

    return covariance, chain
end

"""
    estimate_bounds(chain; width=6, positive=true)

Estimate bounds on parameters in a MarkovChain using a window of size 
`width` multplied by their standard deviation, centered on optimal values.
"""
function estimate_bounds(chain; width=6, absolute_lower_bound=0.0, absolute_upper_bound=Inf)
    # Estimate covariance
    samples = collect_samples(chain)

    Σ = cov(samples, dims=2)

    C₁ = chain[1].param
    C★ = optimal(chain).param

    ParameterType = typeof(C₁).name.wrapper
    bounds = ParameterType(((0.0, 0.0) for c in C₁)...)

    # Set bounds equal to optimal ± width * std(C) for each C.
    for i = 1:length(C₁)
        σ = Σ[i, i]

        lower_bound = C★[i] - width * √σ
        upper_bound = C★[i] + width * √σ

        lower_bound = max(absolute_lower_bound, lower_bound)
        upper_bound = min(absolute_upper_bound, upper_bound)

        bounds[i] = (lower_bound, upper_bound)
    end

    return bounds
end
