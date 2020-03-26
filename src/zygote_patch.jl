using Zygote
using Zygote:@nograd, @adjoint

@nograd gen_samples, canonicalize

@adjoint function free_energy(K::Array, model::MPS, samples)
    free_energy(K, model, samples), function (adjy)
        adjmodel = grad_model(K, model, samples) .* adjy
        return (nothing, adjmodel, nothing)
    end
end
