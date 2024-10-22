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
P = 0.01  # kernel width (probability of error)
x_trn, x_tst, y_trn, y_tst = load_data()
N = ceil(Int64, size(x_trn, 1) / 64)
model = EmpiricalModel(x_trn, y_trn, n_classes, P);

reshape_img(img_flat, shape=(28, 28)) = reshape(img_flat, shape)'
pixel_position(x, y; width=28, height=28) = width*(y-1) + x

function prediction_loop(model::EmpiricalModel, x::SBitSet{N, UInt64}, n_steps) where {N}
    m = SBitSet{N, UInt64}()    # initialize empty mask

    p̂, ŷ = predict(model, x, m)
    println("Step 0 probs $p̂",) 

    for i in 1:n_steps
        # TODO for you: add custom pixel position selection
        x_pos, y_pos = rand(1:28), rand(1:28)
        feature_id = pixel_position(x_pos, y_pos)
        println(feature_id)
        m = push(m, feature_id)         # add feature index to mask

        p̂, ŷ = predict(model, x, m)
        println("Step $i probs $p̂",) 
    end
end

function afa_loop(model::EmpiricalModel, x::SBitSet{N, UInt64}, n_steps) where {N}
    m = SBitSet{N, UInt64}()    # initialize empty mask

    entropy_arr = []

    p̂, ŷ = predict(model, x, m)
    push!(entropy_arr, entropy(p̂))
    println("Step 0 probs $p̂",) 

    for i in 1:n_steps
        feature_id = afa_step(model, x, m)

        m = push(m, feature_id)         # add feature index to mask
        p̂, ŷ = predict(model, x, m)
        push!(entropy_arr, entropy(p̂))
        println("Step $i probs $p̂",) 
    end
    entropy_arr
end

n_steps = 10
observation_id = 2 
x = SBitSet{N, UInt64}(findall(x_tst[:, observation_id] .== 1))   # take the first test set observation
prediction_loop(model, x, n_steps)

entropy_arr = afa_loop(model, x, n_steps)
