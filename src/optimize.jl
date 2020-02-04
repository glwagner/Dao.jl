#
# Optimization tool
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

function optimize(nll, initial_parameters, initial_covariance, 
                  perturbation=NormalPerturbation, perturbation_args...; 
                  samples=100, schedule=nothing, niterations=1, covariance_schedule=nothing)

    initial_link = MarkovLink(nll, initial_parameters)
    covariance_estimate = initial_covariance
    iteration = 1
    chains = []

    while iteration < niterations + 1
        covariance = adjust_covariance_estimate(covariance_schedule, covariance_estimate, iteration)
        sampler = MetropolisSampler(perturbation(covariance, perturbation_args...))

        set_scale!(nll, schedule, iteration, initial_link)

        println("Iterating...")

        wall_time = @elapsed chain = MarkovChain(number_of_samples(samples, iteration), 
                                                     initial_link, nll, sampler)

        @printf("% 24s: %d   \n", "iteration", iteration)
        @printf("% 24s: %d   \n", "samples", length(chain))
        @printf("% 24s: %.3f \n", "acceptance", chain.accceptance)
        @printf("% 24s: %.6f \n", "scaled optimal error", optimal(chain).error / chain[1].error)
        @printf("% 24s: %.6e \n", "unscaled optimal error",optimal(chain).error)
        @printf("% 24s: %.6e \n", "wall time", wall_time)

        # Reset initial links and covariance estimate
        parameter_samples = collect_samples(chain)
        covariance_estimate = cov(parameter_samples, dims=2)
        initial_link = optimal(chain)

        iteration += 1
        push!(chains, chain)
    end

    return covariance_estimate, chains
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

