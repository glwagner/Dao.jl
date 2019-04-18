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
    derror = (test_error - error) / error_scale
    if take_step(d_error)
        new_error = test_error
        new_param = test_param
    else
        new_error = error
        new_param = test_param
    end
    return new_error, new_param, test_param, test_error
end    
    
function markov_chain(loss, param, error, test_param, test_error, error_scale, perturb, nt)
    for i in 1:nt
        param[i+1], error[i+1], test_param[i+1], test_param[i+1] = markov_link(loss, param[i], error[i], error_scale, perturb)
    end
end
    
function take_step(derror)
    u = rand(Uniform(0, 1), 1)
    log_u = log(u[1])
    return log_u < derror
end

end # module
