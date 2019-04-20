using Dao, Test

nice_loss(args...) = 0.0
mean_loss(args...) = Inf

function test_nice_loss()
    link = MarkovLink(nice_loss, [0.0])
    return link.error == 0
end

function test_mean_loss()
    link = MarkovLink(mean_loss, [0.0])
    return link.error == Inf
end

@test test_nice_loss()
@test test_mean_loss()
