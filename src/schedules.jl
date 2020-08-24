abstract type AdaptiveSchedule end

set_iterations!(args...) = nothing
set_iterations!(schedule::AdaptiveSchedule, iterations) = schedule.iterations = iterations

scale(iter, ::Nothing) = 1

set_temperature!(schedule::AdaptiveSchedule, link) = link.error
set_temperature!(::Nothing, link) = nothing

get_temperature(schedule::AdaptiveSchedule, link, previous_chain) = schedule.temperature
get_temperature(::Nothing, link, previous_chain) = optimal(previous_chain).error
get_temperature(::Nothing, link, ::Nothing) = link.error

function adjust_thermostat(annealing_schedule, covariance_schedule, 
                           covariance_estimate, iter, link, previous_chain)

    # Only adapt temperature.
    adapt_schedule!(annealing_schedule, previous_chain)

    # Determine new temperature based on the annealing parameters
    old_temp = get_temperature(annealing_schedule, link, previous_chain)
    new_temp = scale(iter, annealing_schedule) * link.error

    # If acceptance is high enough and iter > 1, set the new covariance based on the
    # previous covariance estimate, new temperature, and covariance schedule parameters.
    new_covariance = (iter == 1 || previous_chain.acceptance > covariance_schedule.acceptance_lower_limit) ? 
                     covariance_estimate : 
                     scale(iter, covariance_schedule) * (new_temp / old_temp) * covariance_estimate

    # Store the temperature for this iteration
    annealing_schedule.temperature = new_temp

    return new_temp, new_covariance
end

"""
    adapt_schedule!(schedule, chain)

Adapt an annealing schedule according to 'acceptance' limits.

If `chain.acceptance` is below `schedule.acceptance_lower_limit`, 
`schedule.convergence_rate` is decreased.

If `chain.acceptance` is above `schedule.acceptance_upper_limit`, 
`schedule.convergence_rate` is increased.

When the schedule governs temperature, this acts to turn up the temperature
when the acceptance is too small.

When the schedule governs covariance, this adaptivity decreases the size of proposal
steps when acceptance is too small.
"""
function adapt_schedule!(schedule::AdaptiveSchedule, chain)
    if chain.acceptance > schedule.acceptance_upper_limit
        schedule.convergence_rate *= schedule.rate_adaptivity
    elseif chain.acceptance < schedule.acceptance_lower_limit
        schedule.convergence_rate /= schedule.rate_adaptivity
    end
    return nothing
end

adapt_schedule!(schedule, chain) = nothing
adapt_schedule!(schedule::AdaptiveSchedule, ::Nothing) = nothing

"""
    mutable struct AdaptiveAlgebraicSchedule{T}

An object for iteratively adapting temperature and covariance during simulated 
annealing with parameters of type `T`.

The keyword arguments to `AdaptiveAlgebraicSchedule`, whose effects are described below, are:

    - `initial_scale`
    - `final_scale`
    - `convergence_rate`
    - `acceptance_upper_limit`
    - `acceptance_lower_limit`
    - `rate_adaptivity`
    - `temperature`

Algebraic decay
====

`AdaptiveAlgebraicSchedule` sets the temperature of a Negative Log Likelihood function 
for an annealing iteration to

    `temperature = scale * error`

and sets the covariance for a proposal probability distribution to

    `covariance = scale .* covariance_estimate`

where `error` is the loss associated with the initial particle position,
`covariance_estimate` is an estimate of the covariance of the negative log
likelihood based on the previous annealing iteration.
`scale` is determined by

    `s = initial_scale + c (iter - 1)^p`

where `p` is `convergence_rate`, `iter` is the iteration number, and `c` is

    `c = (final_scale - initial_scale) / (iterations-1)^p`

where `iterations` is the total number of iterations.

Adaptivity
====

If `chain.acceptance` is above `acceptance_upper_limit`, the `convergence_rate` is
increased. If `chain.acceptance` is below `acceptance_lower_limit`, `convergence_rate`
is decreased.
"""
Base.@kwdef mutable struct AdaptiveAlgebraicSchedule{T} <: AdaptiveSchedule
             initial_scale :: T = 1.0
               final_scale :: T = 1e-3
          convergence_rate :: T = 1.0
                iterations :: T = 0.0
    acceptance_upper_limit :: T = 0.5
    acceptance_lower_limit :: T = 0.05
           rate_adaptivity :: T = 2.0
               temperature :: T = Inf
end

algebraic_scale(n, s¹, sᴺ, p, N) = s¹ + (sᴺ - s¹) * (n-1)^(1/p) / (N-1)^(1/p)

scale(iter, sched::AdaptiveAlgebraicSchedule) = 
    algebraic_scale(iter, sched.initial_scale, sched.final_scale, 
                    sched.convergence_rate, sched.iterations)


"""
    mutable struct AdaptiveExponentialSchedule{T}

An object for iteratively adapting temperature and covariance during simulated 
annealing with parameters of type `T`.

The keyword arguments to `AdaptiveExponentialSchedule`, whose effects are described below, are:

    - `initial_scale`
    - `final_scale`
    - `convergence_rate`
    - `iterations`
    - `acceptance_upper_limit`
    - `acceptance_lower_limit`
    - `rate_adaptivity`

Algebraic decay
====

`AdaptiveExponentialSchedule` uses the function

    `scale(iter, initial_scale, final_scale, niters) = 1 / (1/f₁ + a * exp((n-1) / b)`

where `iter` is the current iteration of the annealing algorithm, 
to scale either temperature or a covariance matrix via

    `temperature = scale * error`

and

    `covariance = scale .* covariance_estimate`

where `error` is the loss associated with the initial particle position,
`covariance_estimate` is an estimate of the covariance of the negative log
likelihood based on the previous annealing iteration.

Adaptivity
====

If `chain.acceptance` is above `acceptance_upper_limit`, the algorithm is perceived 
to be converging too slowly, and `schedule.convergence_rate` is increased by `scale_adaptivity`.

Converly, if `chain.acceptance` is below `acceptance_lower_limit`, `schedule.convergence_rate` is
decreased by `rate_adaptivity`.
"""
Base.@kwdef mutable struct AdaptiveExponentialSchedule{T} <: AdaptiveSchedule
             initial_scale :: T = 1.0
               final_scale :: T = 1e-3
          convergence_rate :: T = 1.0
                iterations :: T = 0.0
    acceptance_upper_limit :: T = 0.5
    acceptance_lower_limit :: T = 0.05
           rate_adaptivity :: T = 2.0
end

scale(iter, schedule::AdaptiveExponentialSchedule) = exponential_scale(iter, schedule.iterations, schedule.initial_scale, 
                                                     schedule.final_scale, schedule.convergence_rate)

coeff(init, final, rate, niters) = exp(-(niters-1)/rate) * (1/final - 1/init)

function exponential_scale(iter, niters, init, final, rate)
    c = coeff(init, final, rate, niters)
    return 1 / (1/init + c * exp((iter-1)/rate))
end


