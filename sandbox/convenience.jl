
#runs ocean_turb model
function model_run(params,ni,nt)
    parameters=KPP.Parameters( CSL = params[1],
       Cτ = params[2],
      CNL = params[3],
    Cstab = params[4],
    Cunst = params[5],
     Cb_U = params[6],
     Cb_T = params[7],
     Cd_U = params[8],
     Cd_T = params[9],
      CRi = params[10],
      CKE = params[11],
       Cn = params[12],
    Cmτ_U = params[13],
    Cmτ_T = params[14],
    Cmb_U = params[15],
    Cmb_T = params[16])
    model = KPP.Model(N=128, L=100.0, stepper=:BackwardEuler,parameters=parameters)
    # Initial condition
    dt = 6
    γ = 0.01
    T₀(z) = 20 + γ*z
    N₀ = sqrt(model.constants.g*model.constants.α*γ)
    # Set T to the function T0(z)
    model.solution.T = T₀
    heat_flux = 75
    temperature_flux = heat_flux / (model.constants.ρ₀ * model.constants.cP)
    buoyancy_flux = temperature_flux * model.constants.g * model.constants.α
    model.bcs.T.top = FluxBoundaryCondition(temperature_flux)
    model.bcs.T.bottom = GradientBoundaryCondition(γ)
    T_p = []
    tmp1 = zeros(128)
    tmp = data(model.solution.T)
    @. tmp1 = tmp
    push!(T_p,tmp1[end:-1:1])
    reset!(model.clock)
    for i in 1:ni
        iterate!(model, dt, nt)
        tmp = data(model.solution.T)
        @. tmp1 = tmp
            push!(T_p,tmp1[end:-1:1])
    end
    return T_p
end


#imports LES data
function load_LES(numimport,pathto)
    #first import LES data
    numimport = 30
    name = pathto*"/convection_"
    top_flux = "75.0_"
    bot_grad = "0.01_"
    grid_points = "128_"
    filenum = "000000100"
    numlist = [lpad("100",9,'0')]
    t = [0]
    for i in 1:numimport
        tmp = string( i * 3600 )
        tmp = lpad(tmp,9,"0")
        push!(numlist,lpad(tmp,9,"0"))
        push!(t, i * 3600 * 6) #in seconds
    end
    T_s = [] #simulation temperature
    filename = name  * top_flux * bot_grad * grid_points * numlist[1] *  ".nc"
    z = ncread(filename,"zC")
    for filenum in numlist
        filename = name  * top_flux * bot_grad * grid_points * filenum *  ".nc"
        #println(filename)
        nc_output = ncread(filename,"T")
        tmp = mean(nc_output,dims=[1,2])[1,1,:] .- 273.15
        push!(T_s,tmp)
    end
return T_s , z , t
end

#Builds Loss Function
function loss_function(params,T_s,ni)
    T_p = model_run(params,ni,3600);
    loss = 0
    for i in 1:(ni+1)
        loss += mean((T_p[i] .- T_s[i] ).^2 )
    end
    loss *= 1 / (ni+1)
    loss==loss ? loss = loss : loss = Inf
    return loss
end



