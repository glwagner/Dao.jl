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

function estimate_covariance(nll, initial_parameters, initial_covariance, perturbation=NormalPerturbation, 
                       perturbation_args...; samples=100, niterations=1)

    initial_link = MarkovLink(nll, initial_parameters)
    covariance = initial_covariance
    iteration = 1
    chains = []

    while iteration < niterations + 1
        sampler = MetropolisSampler(perturbation(covariance, perturbation_args...))

        nll.scale = initial_link.error

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
