module bmnist_pred

using StaticBitSets


# init training data
function init_data(data::AbstractMatrix)
    N = ceil(Int64, size(data, 1) / 64)
	SB = SBitSet{N, UInt64}
    z = zeros(UInt64, N, size(data, 2))
    for i in axes(data, 2)
        sb = SB(findall(data[:,i] .== 1))
        z[:,i] .= sb.pieces
    end
    z    
end
count_unique(y, n_classes) = [count(==(i), y) for i=0:n_classes-1]

struct EmpiricalModel
	z::Matrix{UInt64}
	y::Vector{Int}
	ny::Vector{Int}
	n_classes::Int
	n_features::Int
	P::Float64
	EmpiricalModel(features::AbstractMatrix, labels, n_classes, P) = new(init_data(features), labels, count_unique(labels, n_classes), n_classes, size(features, 1), P)
end

function count_matches(model::EmpiricalModel, x::SBitSet{N,UInt64}, m::SBitSet{N,UInt64}) where {N}
	# computes the number of occurences of (X_o=x_o, Y=y) in train data for all y
	c = zeros(Float64, model.n_classes)

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
		c[y] += exp(tmp)
	end
	c
end

function class_probs(counts, ny)
    lkl = counts ./ ny 			# p(x_o | y)
    prior = ny ./ sum(ny)   # p(y)

    jnt = lkl .* prior      # p(x_o, y)
    jnt / sum(jnt)          # p(y| x_o)
end

function predict(model::EmpiricalModel, x::SBitSet{N,UInt64}, m::SBitSet{N,UInt64}) where {N}
	counts = count_matches(model, x, m)
	probs = class_probs(counts, model.ny)

	pred = argmax(probs)
	probs, pred-1
end

function entropy(p::AbstractArray{T}) where T<:Real
    s = zero(T)
    z = zero(T)
    for i in eachindex(p)
        @inbounds pi = p[i]
        if pi > z
            s += pi * log(pi)
        end
    end
    return -s
end

function afa_objective(counts_0, counts_1, ny)

	py0 = class_probs(counts_0, ny) # p(y|X_i=0, X_o=x_o)
	py1 = class_probs(counts_1, ny) # p(y|X_i=1, X_o=x_o)

	H_0 = entropy(py0)  # H(y|X_i=0, X_o=x_o)
	H_1 = entropy(py1)  # H(y|X_i=1, X_o=x_o)

	jnt_0 = sum(counts_0 ./ sum(ny)) # p(X_i=0, X_o=x_o)
	jnt_1 = sum(counts_1 ./ sum(ny)) # p(X_i=1, X_o=x_o)
	marg = jnt_0 + jnt_1 # p(X_o=x_o)
	# the division by the marginal is not even needed for our purpose
	cond_0 = jnt_0 / marg # p(X_i=0 | X_o=x_o)
	cond_1 = jnt_1 / marg # p(X_i=1 | X_o=x_o)

	cond_0*H_0 + cond_1*H_1
end

function afa_step(model::EmpiricalModel, x::SBitSet{N, UInt64}, m::SBitSet{N, UInt64}) where {N}

	free_vals = setdiff(1:model.n_features, m)
	obj = map(free_vals) do i 
		m_i = push(m, i)
		x_0 =  pop(x, i)
		x_1 = push(x, i)

		counts_0 = count_matches(model, x_0, m_i)
		counts_1 = count_matches(model, x_1, m_i)
		
		afa_objective(counts_0, counts_1, model.ny)
	end
	free_vals[argmin(obj)]
end


export EmpiricalModel, predict, entropy, afa_step

end