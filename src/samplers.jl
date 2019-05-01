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

struct NonNegativeNormalPerturbation{T}
    std :: T
end

const NNNP = NonNegativeNormalPerturbation

(pert::NormalPerturbation)(𝒳) = normal_perturbation(𝒳, pert.std)
(pert::NNNP)(𝒳) = non_negative_normal_perturbation(𝒳, pert.std)

function normal_perturbation(x::AbstractArray, std)
    x_pert = similar(x)
    for i in eachindex(x_pert)
        @inbounds x_pert[i] = x[i] + rand(Normal(0, std[i]))
    end
    return x_pert
end

function non_negative_normal_perturbation(x::AbstractArray, std)
    x_pert = similar(x)
    for i in eachindex(x_pert)
        @inbounds x_pert[i] = max(0, x[i] + rand(Normal(0, std[i])))
    end
    return x_pert
end
