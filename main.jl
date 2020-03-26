push!(LOAD_PATH,"C:/Users/lenovo/Desktop/QML/thermalVQT/MPS/src/")
using mps
using Flux
using Zygote
using StatsBase

model = mps.build_MPS(4,50)

s = mps.gen_samples(model, 5)

K = randn(4,4)

K = (K'+K)/2

mps.free_energy(K, model, s)
l = mps.loss_reinforce(K, model,s)

gradient(l, K, model, s)
