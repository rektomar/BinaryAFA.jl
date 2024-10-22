module bmnist_pred

using StaticBitSets


# init training data
function init_data(data::AbstractMatrix)
    N = ceil(Int64, size(data, 1) / 64)
	SB = SBitSet{N, UInt64}
    z = zeros(UInt64, N, size(data, 2))
    for i in axes(data, 2)
        v = SB(findall(data[:,i] .== 1))
        z[:,i] .= v.pieces
    end
    z    
end
count_unique(y, n_classes) = [count(==(i), y) for i=0:n_classes-1]

struct EmpiricalModel
	z::Matrix{UInt64}
	y::Vector{Int}
	ny::Vector{Int}
	n_classes::Int
	P::Float64
	EmpiricalModel(features, labels, n_classes, P) = new(init_data(features), labels, count_unique(labels, n_classes), n_classes, P)
end

function count_match(model::EmpiricalModel, x::SBitSet{N,UInt64}, m::SBitSet{N,UInt64}) where {N}
	# computes the number of occurences of (X_o=x_o, Y=y) in train data for all y
	h = zeros(Float64, model.n_classes)

    l1p = log(1-model.P) 
    lp = log(model.P)
    D = mapreduce(count_ones, +, m.pieces)  # D = |o|, number of observed features

	@inbounds @simd for col in axes(model.z,2)
		nmatch = 0
		for row in 1:N
			v = ~(model.z[row, col] ⊻ x.pieces[row]) & m.pieces[row]
			nmatch += count_ones(v)
		end
		tmp = l1p * nmatch + lp * (D-nmatch)
		y = model.y[col] + 1
		h[y] += exp(tmp)
	end
	h
end

function class_probs(h, ny)
    lkl = h ./ ny 			# p(x_o | y)
    prior = ny ./ sum(ny)   # p(y)

    jnt = lkl .* prior      # p(x_o, y)
    jnt / sum(jnt)          # p(y| x_o)
end

function predict(model, x, m)
	h = count_match(model, x, m)
	probs = class_probs(h, model.ny)

	pred = argmax(probs)
	probs, pred-1
end

export EmpiricalModel, predict

end