#
# Perturbation functions
#

struct NormalPerturbation{T}
    standard_deviation :: T
end

(perturb::NormalPerturbation)(param) = normal_perturbation(param, perturb.standard_deviation)

function normal_perturbation!(perturbed_x::AbstractArray, x, std)
    for i in eachindex(perturbed_x)
        @inbounds perturbed_x[i] = x[i] + rand(Normal(0, std[i]))
    end
end

function normal_perturbation(x::AbstractArray, std)
    perturbed_x = similar(x)
    normal_perturbation!(perturbed_x, x, std)
    return perturbed_x
end

normal_perturbation(x::Number, std) = x + rand(Normal(0, std))
