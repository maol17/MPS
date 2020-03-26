using StatsBase

function get_lnp(model::MPS, samples::AbstractArray)
    p = zeros(size(samples, 2))
    for i = 1:size(samples, 2)
        A = model.Ar[samples[model.nbits, i] + 1, :, :]
        for j = model.nbits-1:-1:2
            A = model.As[j-1][samples[j, i] + 1, :, :]*A
        end
        A = model.Al[samples[1, i] + 1, :, :]*A
        p[i] = sum(log.(A'A./partition(model)))
    end
    p
end

function unpach_gradient(model::MPS, g)
    (g.Al,g.As...,g.Ar)
end

function get_energy(K::Matrix{T}, samples) where T <: Real
    energy = sum(samples .* (K*samples), dims=1)
end

function free_energy(K::Matrix{T}, model::MPS, samples) where T <: Real
    return mean(get_energy(K, samples) .+ transpose(get_lnp(model, samples)))
end

function loss(K::Matrix{T}, model::MPS, nbatch::Int) where T <: Real
    samples = gen_samples(model, nbatch)
    free_energy(K, model, samples)
end

function loss_reinforce(K::Matrix{T}, model::MPS, samples) where T <: Real
    e = get_energy(K, samples)
    logp = transpose(get_lnp(model, samples))
    f = e .+ logp
    b = mean(f)
    return mean(logp.* (f .- b))
end

function grad_model(K::Matrix{T}, model::MPS, samples) where T <: Real
    model_grad = gradient(loss_reinforce, K, model, samples)[2]
    (model_grad.Al, model_grad.As...,model_grad.Ar)
end
