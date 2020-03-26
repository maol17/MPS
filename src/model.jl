using LinearAlgebra
using Random
using StatsBase

#left canonical
struct MPS{N, T}
    nbits::Int
    As::NTuple{N, Array{T}}
    Al::Array{T}
    Ar::Array{T}
end

function tensor_series(model::MPS)
#    if N>=1
#        return (model.Al, model.As..., model.Ar)
#    else
#        return (model.Al,model.Ar)
#    end
    (model.Al, model.As..., model.Ar)
end

function unpack_gradient(g)
    (g.Al, g.As..., g.Ar)
end

function partition(model::MPS)
    t = tensor_series(model)
    z = t[model.nbits][1,:,:]'*t[model.nbits][1,:,:]+t[model.nbits][2,:,:]'*t[model.nbits][2,:,:]
    z[1]
end

function canonicalize(model::MPS)
    nbits = model.nbits
    t = tensor_series(model)
    tensors = []
    for i = 1:nbits
        push!(tensors, t[i])
    end

    for i in 1:nbits-1
        A = reshape(tensors[i], :, size(tensors[i], 3))
        U, S, V = svd(A)
        tensors[i] = reshape(U, 2, size(tensors[i], 2), :)
        A1 = diagm(S)*V'*tensors[i+1][1, :, :]
        A2 = diagm(S)*V'*tensors[i+1][2, :, :]
        T = zeros(2, size(A1, 1), size(A2, 2))
        T[1, :, :] = A1
        T[2, :, :] = A2
        tensors[i+1] = T
    end

    Al = tensors[1]
    Ar = tensors[nbits]
    As = ntuple(i->tensors[i+1], nbits-2)

    MPS(nbits, As, Al, Ar)
end


function build_MPS(nbits::Int, dim::Int)
    l = randn(2, 1, dim)
    r = randn(2, dim, 1)
    s = ntuple(i->randn(2, dim, dim), nbits-2)

    canonicalize(MPS(nbits,  s, l, r))
end

function gen_samples(model::MPS, nbatch::Int)
    nbits = model.nbits
    z = partition(model)

    sample = zeros(Int, nbits, nbatch)
    p = zeros(Float64, nbits, nbatch)
    #A = NTuple{nbits, NTuple{nbatch, zeros(model.bonddimension)}}
    A = []

    t = tensor_series(model)

    x = []
    for i = 1:nbatch
        p[nbits, i] = ((t[nbits][1, :, :])*(t[nbits][1, :, :])'./z)[1]
        sample[nbits, i] = convert(Int, rand() > p[nbits, i])
        push!(x,t[nbits][sample[nbits, i] + 1, :, :])
    end
    push!(A, x)
    for i = 1:nbits-1
        x = []
        for j = 1:nbatch
            S = t[nbits-i][1, :, :]*A[i][j]
            p[i, j] = (S'S)[1]/((A[i][j])'A[i][j])[1]
            sample[i, j] = convert(Int, rand() > p[i, j])
            push!(x,t[nbits-i][sample[i, j] + 1, :, :]*A[i][j])
        end
        push!(A, x)
    end

    sample
end

function MPS_dispatch!(model::MPS, θ)
    model.Al .= θ[1]
    model.Ar .= θ[model.nbits]
    for i = 2:model.nbits-1
        model.As[i-1] .= θ[i]
    end
    canonicalize(model)
end
