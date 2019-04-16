using Pkg; Pkg.activate(".."); Pkg.instantiate()
using OceanTurb
using BenchmarkTools
using Plots
using Distributions
using Statistics
using Random
using Base
using NetCDF
using JLD2

pathto = "/Users/andresouza/Desktop/Julia/Greg_Wagner/OceanTurb.jl/"
include(pathto*"convenience.jl")

params = [0.1; 0.4; 6.33; 2.0; 6.4; 0.599; 1.36; 0.5; 2.5; 4.32; 0.3; 1.0; 1.0/4; 1.0/2; 1.0/3.0; 1.0/3.0]
T_p = model_run(params,4,3600);

pathtoLES = "/Users/andresouza/Desktop/Julia/Greg_Wagner/OceanTurb.jl/examples/"
pathtoLES = "/home/sandre_google_mit_edu/bucket/free_convection_N128_k1e-4_Float64"
T_s,z,t = load_LES(30,pathtoLES)


Random.seed!(1234) #for reproducibility
num_sim = 3
plist = zeros(num_sim,length(params))
vlist = ones(num_sim) .* 1000
p0 = deepcopy(params)
x = zeros(num_sim)
#set initial value
plist[1,:] = p0
loss_value1  = loss_function(plist[1,:],T_s,4)
val_old = 1.0
vlist[1]  = val_old
eps = 1/loss_value1 #references loss value
#apply MCM algorithm
index_mod = collect(1:8) #choose which indices to modify
#the idea is to make the variance about 50% of the values
sdlist = ones(length(params))
@. sdlist = abs(params[1,:])
xc = zeros(length(params))

for i in 1:(num_sim-1)
    @. xc = plist[i,:]
    for j in index_mod
        xc[j] += rand(Normal(0, sdlist[j]))
    end
    r = rand(Uniform(0,1),1)
    log_r = log(r[1])
    val_new = loss_function(xc,T_s,4) #lets values be negative
    log_ratio = -eps*(val_new - vlist[i])
    log_r < log_ratio ? plist[i+1,:]=copy(xc) : plist[i+1,:] = copy(plist[i,:])
    val_old = val_new
    vlist[i+1] = val_old /loss_value1
    i%20==0 ? println("iteration "*string(i)) : nothing
end

@save "kpp_MCMC.jld" z t T_s plist vlist

