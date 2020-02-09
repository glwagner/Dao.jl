#
# Utilities for optimization
#

function collect_samples(chain)
    parameter_samples = zeros(length(chain.links[1].param), length(chain))
    for (i, link) in enumerate(chain.links)
        @inbounds parameter_samples[:, i] .= link.param
    end
    return parameter_samples
end

number_of_samples(samples::Int, iteration) = samples
number_of_samples(samples::Function, iteration) = samples(iteration)

set_scale!(nll, ::Nothing, iteration, initial_link) = nll.scale = initial_link.error
set_scale!(nll, schedule, iteration, initial_link) = nll.scale = schedule(nll, iteration, initial_link)

adjust_covariance_estimate(::Nothing, estimate, iteration) = estimate
adjust_covariance_estimate(schedule, estimate, iteration) = schedule(estimate, iteration)

"""
    optimize(nll, initial_parameters, initial_covariance, 
             perturbation=NormalPerturbation, perturbation_args...; 
             samples=100, schedule="default", iterations=1, covariance_schedule=nothing)

Attempt to find the optimal parameter value that minimizes a negative log likelihood function `nll` using
simulated-annealing-like procedure.
"""
function optimize(nll, initial_parameters, initial_covariance, 
                  perturbation=NormalPerturbation, perturbation_args...; 
                  samples=100, schedule="default", iterations=1, covariance_schedule=nothing)

    # Default schedule prescribes a linear decrease to temperature equal to error in first link)
    default_schedule(nll, iter, link) = iter/iterations * link.error
    schedule = schedule == "default" ? default_schedule : schedule

    # Initialize optimization run
    initial_link = MarkovLink(nll, initial_parameters)
    covariance_estimate = initial_covariance
    iteration = 1
    chains = []

    while iteration <= iterations
        covariance = adjust_covariance_estimate(covariance_schedule, covariance_estimate, iteration)
        sampler = MetropolisSampler(perturbation(covariance, perturbation_args...))

        set_scale!(nll, schedule, iteration, initial_link)

        println("Annealing...\n")

        wall_time = @elapsed chain = MarkovChain(number_of_samples(samples, iteration), 
                                                 initial_link, nll, sampler)

        # Reset initial links and covariance estimate
        parameter_samples = collect_samples(chain)
        covariance_estimate = cov(parameter_samples, dims=2)
        initial_link = optimal(chain)

        print_optimization_status(iteration, chain, wall_time, covariance_estimate)

        iteration += 1
        push!(chains, chain)
    end

    return covariance_estimate, chains
end

function print_optimization_status(iteration, chain, wall_time, covariance)
    variances = [covariance[i, i] for i = 1:size(covariance)[1]]
    
    @printf("% 24s: %d   \n", "iteration", iteration)
    @printf("% 24s: %d   \n", "samples", length(chain))
    @printf("% 24s: %.6e \n", "wall time", wall_time)
    @printf("% 24s: %.3f \n", "acceptance", chain.acceptance)
    @printf("% 24s: %.3e \n", "temperature", chain.nll.scale)
    @printf("% 24s: %.6f \n", "scaled optimal error", optimal(chain).error / chain[1].error)
    @printf("% 24s: %.6e \n", "unscaled optimal error",optimal(chain).error)
    @printf("% 24s: ", "parameter names"); [@printf("%-8s", n) for n in paramnames(chain[1])]
    @printf("\n")
    @printf("% 24s: ", "variances"); [@printf("%-8.4f", v) for v in variances]
    @printf("\n\n")
    
    return nothing
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
    initialize_bounds(nll, initial_parameters, initial_variance, initial_bounds=nothing; 
                         variance_refinement = 0.1,
                      temperature_refinement = 0.1,
                              search_samples = 100,
                          refinement_samples = 100)

Use a two-step algorithm consisting of an initial search starting from `initial_parameters`,
with `search_samples` and proposals with `initial_variance` and `initial_bounds`, followed by 
a refinement step using `refinement_samples`, the sample bounds as the initial bounds,
a variance equal to `variance_refinement` multiplied by `initial_variance`, and 
a temperature equal to `temperature_refinement` multiplied by the error of the 
optimal search sample. Returns an estimate of parameter bounds and both Markov chains.
If `initial_bounds` are not given, both the initial search and refinement search are 
unbounded.
"""
function initialize_bounds(nll, initial_parameters, initial_variance, initial_bounds=nothing; 
                           variance_refinement = 0.1,
                           temperature_refinement = 0.1,
                           search_samples = 100,
                           refinement_samples = 100
                          )

    # Initialize
    perturbation = initial_bounds === nothing ? NormalPerturbation(initial_variance) :
                                                BoundedNormalPerturbation(initial_variance, initial_bounds)

    sampler = MetropolisSampler(initial_variance, perturbation) 
    initial_link = MarkovLink(nll, initial_parameters)
    initial_chain = MarkovChain(search_samples, initial_link, nll, sampler)

    # Refine
    refined_variance = @. variance_refinement * initial_variance

    perturbation = initial_bounds === nothing ? NormalPerturbation(refinement_variance) :
                                                BoundedNormalPerturbation(refinement_variance, refinement_bounds)

    sampler = MetropolisSampler(refined_variance, perturbation)

    nll.scale = temperature_refinement * optimal(initial_chain).error

    refined_chain = MarkovChain(refinement_samples, optimal(initial_chain).param, nll, sampler)

    # Estimate bounds
    return estimate_bounds(refined_chain), refined_chain, initial_chain
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

    C₀ = chain[1].param
    bounds = [(0.0, 0.0) for c in C₀]

    # Set bounds equal to optimal ± width * std(C) for each C.
    for i = 1:length(C)
        C★ = optimal(chain).param
        σ = Σ[i, i]

        lower_bound = C★ - width * √σ
        upper_bound = C★ + width * √σ

        lower_bound = max(absolute_lower_bound, lower_bound)
        upper_bound = min(absolute_upper_bound, upper_bound)

        bounds[i] = (lower_bound, upper_bound)
    end

    return bounds
end
