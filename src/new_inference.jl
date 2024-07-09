using InteractiveUtils
using BandedMatrices

function TransformerLayerWeights(ggml_dict::Dict{String,Any}, layer_index::Int)
    if !haskey(ggml_dict, "layers.$(layer_index-1).attention.wq.weight")
        error("missing layers.$(layer_index-1) weights")
    end

    # return TransformerLayerWeights(;
    #     rms_att_weight = ggml_dict["layers.$(layer_index-1).attention_norm.weight"],
    #     rms_ffn_weight = ggml_dict["layers.$(layer_index-1).ffn_norm.weight"],
    #     wq             = ggml_dict["layers.$(layer_index-1).attention.wq.weight"],
    #     wk             = ggml_dict["layers.$(layer_index-1).attention.wk.weight"],
    #     wv             = ggml_dict["layers.$(layer_index-1).attention.wv.weight"],
    #     wo             = ggml_dict["layers.$(layer_index-1).attention.wo.weight"],
    #     w1             = ggml_dict["layers.$(layer_index-1).feed_forward.w1.weight"],
    #     w2             = ggml_dict["layers.$(layer_index-1).feed_forward.w2.weight"],
    #     w3             = ggml_dict["layers.$(layer_index-1).feed_forward.w3.weight"],
    # )

    attention = AttentionLayer(
        ggml_dict["layers.$(layer_index-1).attention.wq.weight"],
        ggml_dict["layers.$(layer_index-1).attention.wk.weight"],
        ggml_dict["layers.$(layer_index-1).attention.wv.weight"],
        ggml_dict["layers.$(layer_index-1).attention.wo.weight"],

    )

    attention_rms = AttentionRMSNorm(
        ggml_dict["layers.$(layer_index-1).attention_norm.weight"],
    )

    nn = Chain(
        Dense(ggml_dict["layers.$(layer_index-1).feed_forward.w1.weight"]),
        Dense(ggml_dict["layers.$(layer_index-1).feed_forward.w2.weight"]),
        Dense(ggml_dict["layers.$(layer_index-1).feed_forward.w3.weight"]),
    )

    nn_norm = FFNRMSNorm(
        ggml_dict["layers.$(layer_index-1).ffn_norm.weight"],
    )
    ffn = FFN(nn, nn_norm)

    return TransformerLayer(attention, attention_rms, ffn)
end

# TransformerLayer => TransformerLayerWeights
struct TransformerLayer{A, N, F}
    attention::A
    attention_rms::N
    ffn::F
    # forw
end

Flux.@layer TransformerLayer

struct AttentionLayer{Q,K,V,O,NH,S,D}
    wq::Q
    wk::K
    wv::V
    wo::O
    n_heads::NH
    seq_len::S
    dim::D
    # head_size::HS
end

Flux.@layer AttentionLayer

struct AttentionRMSNorm{W}
    weight::W
end

Flux.@layer AttentionRMSNorm

struct FFN{L, N}
    layers::L
    norm::N
end

Flux.@layer FFN

struct FFNRMSNorm{W}
    weight::W
end

Flux.@layer FFNRMSNorm

# FullModel => TransformerWeights
struct FullModel{T, N, O, TL}
    token_embedding_table::T
    rms_final_weight::N
    output_weight::O
    transformer_layers::TL
end

Flux.@layer FullModel

function TransformerWeights(ggml_dict::Dict{String,Any}, layer_count::Int)
    layers = [TransformerLayerWeights(ggml_dict, 1)]

    for i in 2:layer_count
        push!(layers, TransformerLayerWeights(ggml_dict, i))
    end


    return TransformerWeights(;
        token_embedding_table = ggml_dict["tok_embeddings.weight"],
        rms_final_weight      = ggml_dict["norm.weight"],
        output_weight         = ggml_dict["output.weight"],
        layers,
    )
end

function TransformerWeights(ggml_dict::Dict{String,Any}, layer_count::Int)
    layers = [TransformerLayerWeights(ggml_dict, 1)]

    for i in 2:layer_count
        push!(layers, TransformerLayerWeights(ggml_dict, i))
    end

    token_embedding_table = ggml_dict["tok_embeddings.weight"],
    rms_final_weight      = ggml_dict["norm.weight"],
    output_weight         = ggml_dict["output.weight"],

    return FullModel(
        token_embedding_table,
        rms_final_weight,
        output_weight,
        layers,
    )
end

function Flux.Zygote.ChainRules.rrule(::typeof(reinterpret), ::Type{T}, x::AbstractArray{S}) where {T, S}
    return reinterpret(T, x), Δ -> (Flux.Zygote.ChainRules.NoTangent(), Flux.Zygote.ChainRules.NoTangent(), reinterpret(S, Δ),)
end


training_aware_rope2(wx, pos::Int) = training_aware_rope2(wx, training_aware_pos(wx, pos))
training_aware_rope2(wx, pos::AbstractVector) = rope2(wx, pos)
training_aware_rope2(wx::AbstractMatrix, pos::Int) = rope2(wx, pos)
function rope2(wx, pos::Int)
    cwx = reinterpret(ComplexF32, wx)
    head_size_div2, n_heads = size(cwx)
    
    freq_base = 10000.0f0
    freq_scale = 1.0f0

    theta_scale = freq_base ^ (-inv(Float32(head_size_div2)))
    theta = freq_scale * (pos - 1)

    scales = theta_scale .^ (0:(head_size_div2-1))
    thetas = theta .* scales

    reinterpret(Float32, cwx .* cis.(thetas))
end

training_aware_pos(cwx::AbstractArray{T, 4}, positions) where T = 1:size(cwx, 3)
training_aware_pos(cwx, positions) where T = positions

rope2(wx, ::Nothing) = rope2(wx, 1:size(wx, 3))
function rope2(wx, positions::AbstractVector)

    cwx = reinterpret(ComplexF32, wx)
    head_size_div2, n_heads = size(cwx)

    freq_base = 10000.0f0
    freq_scale = 1.0f0

    theta_scale = freq_base ^ (-inv(Float32(head_size_div2)))
    theta = freq_scale .* (training_aware_pos(cwx, positions) .- 1)

    scales = theta_scale .^ (0:(head_size_div2-1))

    ts = theta' .* scales
    s1, s2 = size(ts)
    thetas = reshape(ts, s1, 1, s2, 1)
    res = reinterpret(Float32, cwx .* cis.(thetas))
    res
end

function (anorm::AttentionRMSNorm)(x)
    s = size(x, 1)
    d = sum(x .^ 2, dims = 1)'
    ss = inv.(sqrt.((d ./ s) .+ 1f-5))

    res = (ss' .* anorm.weight) .* x
    res
end

function (ffnnorm::FFNRMSNorm)(x)
    s = size(x, 1)
    d = sum(x .^ 2, dims = 1)'
    ss = inv.(sqrt.((d ./ s) .+ 1f-5))
    
    (ss' .* ffnnorm.weight) .* x
end

function rmsnorm(x, weight)
    s = size(x, 1)
    d = sum(x .^ 2, dims = 1)'
    ss = inv.(sqrt.((d ./ s) .+ 1f-5))

    (ss' .* weight) .* x
end

silu(x) = x .* sigmoid.(x)

function training_aware_reshape(x, args...)
    if size(x, 2) !== 1
        return reshape(x, args...)
    end
    reshape(x, args[1:2])
end

function training_aware_attention(attn, q::AbstractArray{T, 4}, k, v, args...) where T
    q_permute = permutedims(q, (1, 3, 2, 4))
    k_permute = permutedims(k, (1, 3, 2, 4))
    v_permute = permutedims(v, (1, 3, 2, 4))

    xb, alpha = NNlib.dot_product_attention(q_permute, k_permute, v_permute, nheads = attn.n_heads)
    permutedims(xb, (1, 3, 2, 4))
end

function training_aware_attention(attn, q, k, v, pos, kv, head_size)
    att = attention_weights2(pos, k, q)
    att = att ./ sqrt(Float32(head_size))
    att = Flux.softmax(att, dims = 1)

    Flux.Zygote.ignore() do
        myview = @view kv.value_cache[pos, :, :]
        copyto!(myview, v)
    end

    xb = combine_values2(kv.value_cache, att)
end

function (attn::AttentionLayer)(x, kv, pos)
    # seq_len, batch_size = size(x)

    head_size = attn.dim ÷ attn.n_heads
    q = attn.wq * x
    k = attn.wk * x
    v = attn.wv * x

    q_reshaped = training_aware_reshape(q, head_size, attn.n_heads, attn.seq_len, :) # head_size, n_heads, seq_len, batch_size
    k_reshaped = training_aware_reshape(k, head_size, attn.n_heads, attn.seq_len, :)
    v_reshaped = training_aware_reshape(v, head_size, attn.n_heads, attn.seq_len, :)

    # apply RoPE rotation to the q and k vectors for each head
    q_ = training_aware_rope2(q_reshaped, pos)
    k_ = training_aware_rope2(k_reshaped, 1:pos)

    xb = training_aware_attention(attn, q_, k_, v_reshaped, pos, kv, head_size)

    # final matmul to get the output of the attention
    xb2 = attn.wo * reshape(xb, attn.dim, :)
    xb2
end

function training_aware_attention_slice(q, pos)
    1:pos
end
function training_aware_attention_slice(q::AbstractArray{T, 4}, pos) where T
    1:size(q, 3)
end

attention_weights2(att, key_cache, q) = attention_weights2(size(att, 1), key_cache, q)
function attention_weights2(pos::Int, key_cache, q)
    slicedims = training_aware_attention_slice(q, pos)
    c = map(eachslice(key_cache[:,:,slicedims,:], dims = 3)) do x
        f..., batch_dim = size(x)
        x_ = reshape(x, f..., 1, batch_dim)
        sum(q .* x_, dims = 1)
    end
    reduce(vcat, c)
end

# combine_values2(xb, value_cache, att) = combine_values2(copy(xb), xb, value_cache, att)
function combine_values2(value_cache, att)
    t = size(att, 1)

    c = map(x -> sum(att .* x, dims = 1), eachslice(value_cache[1:t, :, :], dims = 2))

    reduce(vcat, c)
end

(tr::TransformerLayer)(args) = tr(args...)
function (layer::TransformerLayer)(x, kv, pos, idx)
    # attention rmsnorm
    xb = layer.attention_rms(x)
    
    kv_layer = kv[idx]
    att = layer.attention(xb, kv_layer, pos)
    # residual connection back into x
    x2 = x .+ att

    # ffn rmsnorm
    xb_norm = layer.ffn.norm(x2)
    
    # F.silu silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
    # Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
    # first calculate self.w1(x) and self.w3(x)
    hb = silu.(layer.ffn.layers[1](xb_norm))
    hb2 = layer.ffn.layers[3](xb_norm)
    
    # final matmul to get the output of the ffn
    xb_out = layer.ffn.layers[2](hb .* hb2)

    # residual connection
    (x2 .+ xb_out, kv, pos)
end

function (t::NewTransformerModel)((x, kv, pos))
    for (idx, layer) in enumerate(t.chain[1:1])
        x, kv, pos = layer((x, kv, pos, idx))
    end
    (x, kv, pos)
end

function (fm::FullModel)(x, kv, pos)
    x_emb = fm.token_embedding_table(x)
    transformer = fm.transformer_layers
    out, _, _ = transformer((x_emb, kv, pos))
    x_norm = rmsnorm(out, fm.rms_final_weight)

    logits = fm.output_weight' * x_norm
end

@views function transformer!(token::Int, pos::Int, config::ModelConfig, s::RunState, model::FullModel)
    x = s.x

    (;
        dim,
        hidden_dim,
        n_layers,
        n_heads,
    ) = config

    head_size = dim ÷ n_heads

    # copy the token embedding into x
    dequantize!(x, model.token_embedding_table.weight[:, token])

    transformer = model.transformer_layers
    kv = s.kvcache_layers

    # forward all the layers
    x, _, _ = transformer((x, kv, pos))

    # final rmsnorm
    x_norm = rmsnorm(x, model.rms_final_weight)

    # classifier into logits
    logits = model.output_weight' * x_norm
    copyto!(s.logits, logits)

    return nothing
end