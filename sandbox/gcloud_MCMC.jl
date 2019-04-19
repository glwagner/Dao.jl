using Pkg
using JLD2
pathtoDao = "../src/Dao.jl"
include(pathtoDao)
using .Dao
pathtoOceanTurb = "../src/OceanTurb"
include(pathtoDao)
using .OceanTurb
using BenchmarkTools
using Plots
using Distributions
using Statistics
using Random
using Base
using NetCDF

#import LES
include("convenience.jl")
pathtoLES = "/Users/andresouza/Desktop/Julia/Greg_Wagner/OceanTurb.jl/examples/"
T_s ,z ,t = load_LES(30,pathtoLES);

#Builds Loss Function
ni = 4
function loss(params)
    T_p = model_run(params,ni,36);
    loss = 0
    for i in 1:(ni+1)
        loss += mean((T_p[i] .- T_s[i] ).^2 )
    end
    loss *= 1 / (ni+1)
    if isnan(loss)
        loss = Inf   #if NaNs are produced output infinity for the loss function
    end
    return loss
end
#function to keep parameters between 0 and 1
torus(x) = (x)%1 - 0.5 * (sign(x) - 1) 
#function to keep parameters between a and b, where b>a
torus(x,a,b) = ((x)%1 - 0.5 * (sign(x) - 1) )*(b-a) + a
#now we can define a random walk
params = [0.1; 0.4; 6.33; 2.0; 6.4; 0.599; 1.36; 0.5; 2.5; 0.3; 4.32; 1.0; 1.0/4; 1.0/2; 1.0/3.0; 1.0/3.0];
#can flip 10 and 11 for some reason
index_mod = [1 3 6 7 10 11]
sdlist = ones(length(params))
@. sdlist = abs(params[1,:])
#define a random walk
function perturbation(params)
    new_params = deepcopy(params)
    for i in index_mod
        new_params[i] += rand(Normal(0, sdlist[i]))
    end
    new_params[1]  = torus(new_params[1])
    new_params[10]  = torus(new_params[10])
    return new_params
end
Random.seed!(1234)
params = [0.1; 0.4; 6.33; 2.0; 6.4; 0.599; 1.36; 0.5; 2.5; 0.3; 4.32; 1.0; 1.0/4; 1.0/2; 1.0/3.0; 1.0/3.0];
error_scale = 0.001
Dao.markov_chain_with_save(loss, params, perturbation, error_scale, 10, "test");