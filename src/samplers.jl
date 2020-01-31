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
    bounds :: B
    function BoundedNormalPerturbation(covariance::AbstractArray, bounds)
        distribution = MvNormal(covariance)
        return new{typeof(distribution), typeof(bounds)}(distribution, bounds)
    end
end

(pert::NormalPerturbation)(θ) = normal_perturbation(θ, pert.distribution)
(pert::BoundedNormalPerturbation)(θ) = bounded_normal_perturbation(θ, pert.distribution, pert.bounds)

torus(x, lower, upper) = lower + rem(x - lower, upper - lower, RoundDown)

normal_perturbation(θ::AbstractArray, distribution) = θ .+ rand(distribution)
    
function bounded_normal_perturbation(θ, distribution, bounds)
    θ′ = normal_perturbation(θ, distribution)
    return torus.(θ′, bounds[1], bounds[2])
end
