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

function count_match(z::Matrix{UInt64}, x::SBitSet{N,UInt64}, m::SBitSet{N,UInt64}, labels, P) where {N}
	# computes the number of occurences of (X_o=x_o, Y=y) in train data for all y
	n_classes = 10
	h = zeros(Float64, n_classes)

    l1p = log(1-P) 
    lp = log(P)
    D = mapreduce(count_ones, +, m.pieces)  # D = |o|, number of observed features

	@inbounds @simd for col in axes(z,2)
		nmatch = 0
		for row in 1:N
			v = ~(z[row, col] ⊻ x.pieces[row]) & m.pieces[row]
			nmatch += count_ones(v)
		end
		tmp = l1p * nmatch + lp * (D-nmatch)
		y = labels[col] + 1
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

function predict(data, labels, x, m, ny, P)
	h = count_match(data, x, m, labels, P)
	probs = class_probs(h, ny)

	pred = argmax(probs)
	probs, pred-1
end


export init_data, count_match, classify, predict

end