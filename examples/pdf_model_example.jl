using Pkg; Pkg.activate("..")

struct PDFModel
    pdf :: Function
end

struct TrivialData end

# define an initial probability distribution
p(x) = (exp(-(x-2)^2/2) + exp(-(x+2)^2/2)) / (2*sqrt(2pi))
model = PDFModel(p)
data = TrivialData()

# define the loss function as the log of the probability distribution
compute_loss(x, model, data) = -log(model.p(x))

loss = LossFunction(model, data, compute_loss, 1)

# define a random walk
function perturbation(params)
    new_params = deepcopy(params)
    for i in 1:length(params)
        new_params[i] += rand(Normal(0, 1))
    end
    return new_params
end
