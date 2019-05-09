struct MetropolisSampler{T}
    perturb :: T
end

NormalMetropolisSampler(std) = MetropolisSampler(NormalPerturbation(std))

#
# Perturbation functions
#

torus(x, lower, upper) = lower + (x % 1 - 0.5 * (sign(x) - 1)) * (upper - lower)
bound(x, lower::Number, upper::Number) = torus(x, upper, lower)

struct NormalPerturbation{T}
    std :: T
end

struct BoundedNormalPerturbation{T, B}
    std :: T
    bounds :: B
end

(pert::NormalPerturbation)(θ) = normal_perturbation(θ, pert.std)
(pert::BoundedNormalPerturbation)(θ) = bounded_normal_perturbation(θ, pert.std, pert.bounds)

function normal_perturbation(θ::AbstractArray, std)
    θ′ = similar(θ)
    for i in eachindex(θ′)
        @inbounds θ′[i] = θ[i] + rand(Normal(0, std[i]))
    end
    return θ′
end

function bounded_normal_perturbation(θ, std, bounds)
    θ′ = similar(θ)
    for i in eachindex(θ′)
        @inbounds θ′[i] = torus(θ[i] + rand(Normal(0, std[i])), bounds[i][1], bounds[i][2])
    end
    return θ′
end
