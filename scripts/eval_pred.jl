using bmnist_pred

using StaticBitSets
using Statistics
using DelimitedFiles
using NPZ

function load_data()
    x_trn = npzread("data/bmnist_trn_x.npy") |> permutedims;
    y_trn = npzread("data/bmnist_trn_y.npy");

    x_tst = npzread("data/bmnist_tst_x.npy") |> permutedims;        
    y_tst = npzread("data/bmnist_tst_y.npy");
    x_trn, x_tst, y_trn, y_tst
end

n_classes = 10
P = 0.01
x_trn, x_tst, y_trn, y_tst = load_data()
N = ceil(Int64, size(x_trn, 1) / 64)
model = EmpiricalModel(x_trn, y_trn, n_classes, P);


function prediction_loop(model::EmpiricalModel, x::SBitSet{N, UInt64}, n_steps) where {N}
    m = SBitSet{N, UInt64}()    # initialize empty mask

    p̂, ŷ = predict(model, x, m)
    println("Step 0 probs $p̂",) 

    for i in 1:n_steps
        id = rand(1:784)
        m = push(m, id)         # add feature index to mask

        p̂, ŷ = predict(model, x, m)
        println("Step $i probs $p̂",) 
    end
end

n_steps = 10
x = SBitSet{N, UInt64}(findall(x_tst[:, 1] .== 1))   # take the first test set observation
prediction_loop(model, x, n_steps)
