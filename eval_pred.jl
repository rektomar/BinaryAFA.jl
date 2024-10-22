using bmnist_pred

using StaticBitSets
using Statistics
using DelimitedFiles
using NPZ

ids = [400, 401, 402]
nsteps = length(ids)
P = 0.01

function load_data()
    n_classes = 10
    x_trn = npzread("data/bmnist_trn_x.npy") |> permutedims;
    y_trn = npzread("data/bmnist_trn_y.npy");

    x_tst = npzread("data/bmnist_tst_x.npy") |> permutedims;        
    y_tst = npzread("data/bmnist_tst_y.npy");
    ny = [count(==(i), y_trn) for i=0:n_classes-1]
    x_trn, x_tst, y_trn, y_tst, ny
end

function prediction_loop(x::SBitSet{N, UInt64}, z, y_trn, ny, P) where {N}
    m = SBitSet{N, UInt64}()    # initialize empty mask

    for i in 1:nsteps
        id = ids[i]
        m = push(m, id)         # add feature index to mask

        p̂, ŷ = predict(z, y_trn, x, m, ny, P)
        println(p̂) 
    end
end

x_trn, x_tst, y_trn, y_tst, ny = load_data()
z = init_data(x_trn)
x = SBitSet{13, UInt64}(findall(x_tst[:, 1] .== 1))   # take the first test set observation

prediction_loop(x, z, y_trn, ny, P)
