struct MetropolisSampler{T}
    perturb :: T
end

NormalMetropolisSampler(std) = MetropolisSampler(NormalPerturbation(std))

#
# Perturbation functions
#

struct NormalPerturbation{T}
    std :: T
end

(pert::NormalPerturbation)(𝒳) = normal_perturbation(𝒳, pert.std)

function normal_perturbation!(x_pert::AbstractArray, x, std)
    for i in eachindex(x_pert)
        @inbounds x_pert[i] = x[i] + rand(Normal(0, std[i]))
    end
end

function normal_perturbation(x::AbstractArray, std)
    x_pert = similar(x)
    normal_perturbation!(x_pert, x, std)
    return x_pert
end
