module Dao

export
    markov_link,
    markov_chain

using 
    Distributions,
    Statistics,
    Random

function markov_link(loss, param, error, error_scale, perturb)
    test_param = perturb(param)
    test_error = loss(test_param)
    d_error = (test_error - error) / error_scale
    if take_step(d_error)
        new_error = test_error
        new_param = test_param
    else
        new_error = error
        new_param = param
    end
    return new_param, new_error, test_param, test_error
end    
    
function markov_chain!(loss, param, error, test_param, test_error, error_scale, perturb, nt)
    for i in 1:nt
        param[i+1], error[i+1], test_param[i+1], test_error[i+1] = markov_link(loss, param[i], error[i], error_scale, perturb)
    end
end
    
function markov_chain(loss, init_param, perturb, error_scale, nt)
    param = ones(length(init_param),nt+1)
    @views @. param[:,1] = init_param
    test_param = deepcopy(param) 
    error = ones(nt+1) .* 10^6
    test_error = deepcopy(error)
    error[1] = loss(init_param)/error_scale
    for i in 1:nt
        @views tmp1,tmp2,tmp3,tmp4 = markov_link(loss, param[:,i], error[i], error_scale, perturb)
        @views @. param[:,i+1] = tmp1
        error[i+1] = tmp2
        @views @. test_param[:,i+1] = tmp3 
        test_error[i+1] = tmp4
    end
    return param, error
end
    
function markov_chain_extra(loss, init_param, perturb, error_scale, nt)
    param = ones(length(init_param),nt+1)
    @views @. param[:,1] = init_param
    test_param = deepcopy(param) 
    error = ones(nt+1) .* 10^6
    test_error = deepcopy(error)
    error[1] = loss(init_param)/error_scale
    for i in 1:nt
        @views tmp1,tmp2,tmp3,tmp4 = markov_link(loss, param[:,i], error[i], error_scale, perturb)
        @views @. param[:,i+1] = tmp1
        error[i+1] = tmp2
        @views @. test_param[:,i+1] = tmp3 
        test_error[i+1] = tmp4
    end
    return param, error, test_param, test_error
end
    
function take_step(derror)
    u = rand(Uniform(0, 1), 1)
    log_u = log(u[1])
    return log_u < derror
end

end # module
