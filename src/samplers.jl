struct MetropolisSampler{T}
    perturb :: T
end

NormalMetropolisSampler(std) = MetropolisSampler(NormalPerturbation(std))

#
# Perturbation functions
#

struct NormalPerturbation{T}
    distribution :: T
    function NormalPerturbation(covariance::AbstractArray)
        distribution = MvNormal(covariance)
        return new{typeof(distribution)}(distribution, bounds)
    end
end

struct BoundedNormalPerturbation{T, B}
    distribution :: T
    lower_bounds :: B
    upper_bounds :: B
    function BoundedNormalPerturbation(covariance::AbstractArray, bounds)
        distribution = MvNormal(covariance)
        lower_bounds = [b[1] for b in bounds]
        upper_bounds = [b[2] for b in bounds]
        return new{typeof(distribution), typeof(lower_bounds)}(distribution, lower_bounds, upper_bounds)
    end
end

@inline (pert::NormalPerturbation)(θ) = normal_perturbation(θ, pert.distribution)

@inline (pert::BoundedNormalPerturbation)(θ) =
    bounded_normal_perturbation(θ, pert.distribution, pert.lower_bounds, pert.upper_bounds)

@inline torus(x, lower, upper) = lower + rem(x - lower, upper - lower, RoundDown)

@inline function normal_perturbation(θ, distribution) 
    θ′ = similar(θ)
    θ′ .= θ .+ rand(distribution)
    return θ′
end
    
@inline function bounded_normal_perturbation(θ, distribution, lower_bounds, upper_bounds)
    θ′ = normal_perturbation(θ, distribution)
    θ′ .= torus.(θ′, lower_bounds, upper_bounds)
    return θ′
end
