using InteractiveUtils
using BandedMatrices

function Flux.Zygote.ChainRules.rrule(::typeof(reinterpret), ::Type{T}, x::AbstractArray{S}) where {T, S}
    return reinterpret(T, x), Δ -> (Flux.Zygote.ChainRules.NoTangent(), Flux.Zygote.ChainRules.NoTangent(), reinterpret(S, Δ),)
end

function rope2(wx, pos::Int)
    global gwx = wx
    cwx = reinterpret(ComplexF32, wx)
    # cwx = ComplexF32.(wx)
    head_size_div2, n_heads = size(cwx)
    
    freq_base = 10000.0f0
    freq_scale = 1.0f0

    theta_scale = freq_base ^ (-inv(Float32(head_size_div2)))
    theta = freq_scale * (pos - 1)

    # @show theta
    # theta_mat = zeros(head_size_div2, n_heads)
    # scales = [theta ; theta_scale .^ (1:head_size_div2-1)]
    scales = theta_scale .^ (0:(head_size_div2-1))
    thetas = theta .* scales

    # @show theta
    # @show size(wx)
    # @show size(thetas)
    reinterpret(Float32, cwx .* cis.(thetas))
    # thetas
end

function rope2(wx, positions::AbstractVector)
    # @show size(wx)
    pes = map(pos -> rope2(wx, pos), positions)
    # @show size.(pes)
    cat(pes..., dims = 3)
end

function (anorm::AttentionRMSNorm)(x)
    ss = inv.(sqrt.((dot(x, x) ./ length(x)) .+ 1f-5))
    anorm.weight .* ss .* x
end

function (ffnnorm::FFNRMSNorm)(x)
    ss = inv.(sqrt.((dot(x, x) ./ length(x)) .+ 1f-5))
    ffnnorm.weight .* ss .* x
end

function rmsnorm(x, weight)
    ss = inv.(sqrt.((dot(x, x) ./ length(x)) .+ 1f-5))
    weight .* ss .* x
end

silu(x) = x .* sigmoid.(x)

function (attn::AttentionLayer)(x, kv, pos::Int)
    q = attn.wq' * x
    k = attn.wk' * x
    v = attn.wv' * x

    # @show size(x)
    
    head_size, n_heads = (64, 8) # attn.head_size, attn.n_heads # (64, 8)
    q_reshaped = reshape(q, head_size, n_heads)
    k_reshaped = reshape(k, head_size, n_heads)
    
    # apply RoPE rotation to the q and k vectors for each head
    q_ = rope2(q_reshaped, pos)
    k_ = rope2(k_reshaped, 1:pos)
    # global gk2 = k_
    # @show size(k_)
    # @show size(kv.key_cache)

    # needs to get the k from previous 1:pos time steps
    # copyto!(kv.key_cache[:, :, pos], k_)
    # all_keys = rope2(kv.key_cache, 1:pos)
    # @show size(all_keys)
    # att = reshape(s.att[1:(n_heads*pos)], pos, n_heads)

    # multihead attention
    att = attention_weights2(pos, k_, q_)
    att = att ./ sqrt(Float32(head_size))
    att = Flux.softmax(att)



    myview = @view kv.value_cache[pos, :, :]
    copyto!(myview, v)
    # @show sum(kv.value_cache)

    # weighted sum of the values
    xb = combine_values2(kv.value_cache, att)

    # @show sum(xb)

    # final matmul to get the output of the attention
    # matmul!(s.xb2, w.wo, s.xb)

    global gv_new = v
    global gatt_new = att
    global gxb_new = xb

    # @show size(attn.wo)
    xb2 = attn.wo' * vec(xb)

    global gxb2_new = xb2
    # @show sum(xb2)
    xb2

    # att, v

    # q_, k_, v
end

attention_weights2(att, key_cache, q) = attention_weights2(size(att, 1), key_cache, q)
function attention_weights2(pos::Int, key_cache, q)
    # size(att) = pos x n_heads
    t = pos # size(att, 1)
    # dot product over attention span
    c = map(x -> sum(q .* x, dims = 1), eachslice(key_cache[:,:,1:t], dims = 3))
    reduce(vcat, c)
end

# combine_values2(xb, value_cache, att) = combine_values2(copy(xb), xb, value_cache, att)
function combine_values2(value_cache, att)
    t = size(att, 1)

    c = map(x -> sum(att .* x, dims = 1), eachslice(value_cache[1:t, :, :], dims = 2))

    reduce(vcat, c)
end

(tr::TransformerLayer)(args) = tr(args...)
function (layer::TransformerLayer)(x, kv, pos)
    # attention rmsnorm
    xb = layer.attention_rms(x)

    xb2 = layer.attention(xb, kv, pos)

    # residual connection back into x
    x2 = x .+ xb2

    # ffn rmsnorm
    # rmsnorm!(s.xb, x, w.rms_ffn_weight)
    xb_norm = layer.ffn.norm(x2)
    
    # F.silu silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
    # Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
    # first calculate self.w1(x) and self.w3(x)
    hb = silu.(layer.ffn.layers[1](xb_norm))
    hb2 = layer.ffn.layers[3](xb_norm)
    
    # final matmul to get the output of the ffn
    xb_out = layer.ffn.layers[2](hb .* hb2)

    # @show sum(xb_out)
    # pos == 1 && error()

    # residual connection
    (x .+ xb_out, kv, pos)
end

@views function transformer!(token::Int, pos::Int, config::ModelConfig, s::RunState, model::FullModel)
    global gs = s
    x = s.x

    (;
        dim,
        hidden_dim,
        n_layers,
        n_heads,
    ) = config

    head_size = dim ÷ n_heads

    # copy the token embedding into x
    dequantize!(x, model.token_embedding_table[:, token])

    transformer = model.transformer_layers
    kv = s.kvcache_layers[1]

    # forward all the layers
    x, _, _ = transformer((x, kv, pos))

    # final rmsnorm
    x_norm = rmsnorm(x, model.rms_final_weight)

    # classifier into logits
    logits = model.output_weight' * x_norm
    copyto!(s.logits, logits)
    # @show sum(s.logits)

    return nothing
end