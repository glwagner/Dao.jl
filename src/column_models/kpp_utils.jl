#
# Parameter sets
#

const nfcparams = 6

mutable struct FreeConvectionParameters{T} <: FieldVector{nfcparams, T}
     CRi :: T
     CNL :: T
     CKE :: T
      Cτ :: T
    Cb_U :: T
    Cb_T :: T
end

sample_params = FreeConvectionParameters((0 for i = 1:nfcparams)...)

function FreeConvectionParameters()
    parameters = KPP.Parameters()
    p_vary = (getproperty(parameters, name) for name in propertynames(sample_params))
    FreeConvectionParameters(p_vary...)
end

function KPP_Parameters(p, K₀=1e-4)
    param_dict = Dict((k, getproperty(params, k)) for k in propertynames(params))
    param_dict[:K₀] = K₀
    return KPP.Parameters(; param_dict...)
end

#
# Loss functions
#

"""
    temperature_loss(params, model, data; iters=1)

Compute the error between model and data for one iteration of
the `model.turbmodel`.
"""
function temperature_loss(params, model, data; iters=1)
    kpp_parameters = KPP_Parameters(params, data.K₀)
    turbmodel = model.turbmodel
    turbmodel.parameters = kpp_parameters

    # Set initial condition as first non-trivial time-step
    turbmodel.solution.U = data.U[2]
    turbmodel.solution.V = data.V[2]
    turbmodel.solution.T = data.T[2]

    loss = 0.0

    for i = 3:3+iters
        run_until!(turbmodel, model.dt, data.t[i])
        T_model = OceanTurb.data(turbmodel.solution.T)
        loss += mean(data.T[i].^2 .- T_model.^2)
    end

    return loss
end

#
# Models
#

"""
    simple_flux_model(constants; N=40, L=400, Bz=0.01, Fb=1e-8, Fu=0,
                      parameters=KPP.Parameters())

Construct a model forced by 'simple', constant atmospheric buoyancy flux `Fb`
and velocity flux `Fu`, with resolution `N`, domain size `L`, and
and initial linear buoyancy gradient `Bz`.
"""
function simple_flux_model(constants; N=40, L=400, Bz=0.01, Fb=1e-8, Fu=0,
                           parameters=KPP.Parameters())

    model = KPP.Model(N=N, L=L, parameters=parameters, constants=constants,
                      stepper=:BackwardEuler)

    # Initial condition
    Tz = model.constants.α * model.constants.g * Bz
    T₀(z) = 20 + Tz*z
    model.solution.T = T₀

    # Fluxes
    Fθ = Fb / (model.constants.α * model.constants.g)
    model.bcs.U.top = FluxBoundaryCondition(Fu)
    model.bcs.T.top = FluxBoundaryCondition(Fθ)
    model.bcs.T.bottom = GradientBoundaryCondition(Tz)

    return model
end

function simple_flux_model(datapath::AbstractString)
    data_params, constants_dict = get_data_params(datapath)
    constants = KPP.Constants(; constants_dict...)
    simple_flux_model(constants, parameters; data_params...)
end

function simple_flux_model(data::ColumnModelData)
    constants = KPP.Constants(; α=data.α, g=data.g, f=data.f)
    setup = Dict((p, getproperty(data, p)) for p in (:N, :L, :Bz, :Fb, :Fu))
    return simple_flux_model(constants; setup...)
end
