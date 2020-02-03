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

function optimize(nll, initial_parameters, initial_covariance, 
                  perturbation=NormalPerturbation, perturbation_args...; 
                  samples=100, schedule=nothing, niterations=1)
                  

    initial_link = MarkovLink(nll, initial_parameters)
    covariance = initial_covariance
    iteration = 1
    chains = []

    while iteration < niterations + 1
        sampler = MetropolisSampler(perturbation(covariance, perturbation_args...))

        set_scale!(nll, schedule, iteration, initial_link)

        wall_time = @elapsed new_chain = MarkovChain(number_of_samples(samples, iteration), initial_link, nll, sampler)

        push!(chains, new_chain)

        parameter_samples = collect_samples(new_chain)

        @printf("Iteration: %d, wall time: %.4f s, scaled optimal error: %.6f, unscaled optimal error: %.6e\n", 
                iteration, wall_time, optimal(new_chain).error / nll.scale, optimal(new_chain).error)

        covariance = cov(parameter_samples, dims=2)

        initial_link = optimal(new_chain)

        iteration += 1
    end

    return covariance, chains
end

function estimate_covariance(nll, initial_parameters, initial_covariance,
                             perturbation=NormalPerturbation, perturbation_args...; 
                             samples=100, schedule=nothing)

    initial_link = MarkovLink(nll, initial_parameters)
    nll.scale = initial_link.error

    covariance = initial_covariance

    sampler = MetropolisSampler(perturbation(covariance, perturbation_args...))
    chain = MarkovChain(samples, initial_link, nll, sampler)
    parameter_samples = collect_samples(new_chain)
    covariance = cov(parameter_samples, dims=2)

    return covariance, chain
end

