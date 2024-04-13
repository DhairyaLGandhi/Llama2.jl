const LLAMA_FILE_MAGIC_GGJT = 0x67676a74
const LLAMA_FILE_VERSION_GGJT_V2 = 3

@enum GGML_TYPE begin
    GGML_TYPE_F32 = 0
    GGML_TYPE_Q4_K = 12
    GGML_TYPE_Q6_K = 14
end

struct GGMLHeader
    n_vocab::Int
    n_embd::Int
    n_mult::Int
    n_head::Int
    n_layer::Int
    n_rot::Int
end

function read_ggml_header(file::IOStream)
    header = GGMLHeader(
        read(file, UInt32),
        read(file, UInt32),
        read(file, UInt32),
        read(file, UInt32),
        read(file, UInt32),
        read(file, UInt32),
    )

    ftype = read(file, UInt32) # FIXME

    return header
end

function read_ggml_tokenizer(file::IOStream, n_vocab::Int)
    id_to_token = Vector{String}(undef, n_vocab)
    token_to_id = Dict{String,Int}()
    token_scores = Vector{Float32}(undef, n_vocab)

    for i in 1:n_vocab
        len = Int(read(file, UInt32))
        word = String(read(file, len))

        score = read(file, Float32)

        id_to_token[i] = word
        token_scores[i] = score

        token_to_id[word] = i
    end

    return BPETokenizer(id_to_token, token_to_id, token_scores)
end

function read_ggml_tensor(tensor_type::GGML_TYPE, size::Tuple, file::IOStream)
    N = length(size)

    if tensor_type == GGML_TYPE_F32
        tensor = Array{Float32,N}(undef, size)
    elseif tensor_type == GGML_TYPE_Q4_K
        @assert size[1] % QK_K == 0
        size = (size[1] ÷ QK_K, size[2:end]...)
        tensor = Array{block_q4_K,N}(undef, size)
    elseif tensor_type == GGML_TYPE_Q6_K
        @assert size[1] % QK_K == 0
        size = (size[1] ÷ QK_K, size[2:end]...)
        tensor = Array{block_q6_K,N}(undef, size)
    else
        error("tensor type $tensor_type not implemented")
    end

    read!(file, tensor)
    return tensor
end

function read_ggml_tensor_dict(file::IOStream; show_progress=true)
    if show_progress
        pos = position(file)
        seekend(file)
        file_size = position(file)
        seek(file, pos)

        progress = Progress(file_size, "Loading model...")
        update!(progress, position(file))
    end

    tensor_dict = Dict{String,Any}()

    while !eof(file)
        n_dims = Int(read(file, UInt32))
        name_len = Int(read(file, UInt32))
        tensor_type = GGML_TYPE(read(file, UInt32))

        size = [Int(read(file, UInt32)) for _ in 1:n_dims]
        size = (size...,)

        name = String(read(file, name_len))

        # skip to the next multiple of 32 bytes
        pos = position(file)
        seek(file, pos + (-pos & 31))

        tensor_dict[name] = read_ggml_tensor(tensor_type, size, file)

        show_progress && update!(progress, position(file))
    end

    return tensor_dict
end

function TransformerLayerWeights(ggml_dict::Dict{String,Any}, layer_index::Int)
    if !haskey(ggml_dict, "layers.$(layer_index-1).attention.wq.weight")
        error("missing layers.$(layer_index-1) weights")
    end

    return TransformerLayerWeights(;
        rms_att_weight = ggml_dict["layers.$(layer_index-1).attention_norm.weight"],
        rms_ffn_weight = ggml_dict["layers.$(layer_index-1).ffn_norm.weight"],
        wq             = ggml_dict["layers.$(layer_index-1).attention.wq.weight"],
        wk             = ggml_dict["layers.$(layer_index-1).attention.wk.weight"],
        wv             = ggml_dict["layers.$(layer_index-1).attention.wv.weight"],
        wo             = ggml_dict["layers.$(layer_index-1).attention.wo.weight"],
        w1             = ggml_dict["layers.$(layer_index-1).feed_forward.w1.weight"],
        w2             = ggml_dict["layers.$(layer_index-1).feed_forward.w2.weight"],
        w3             = ggml_dict["layers.$(layer_index-1).feed_forward.w3.weight"],
    )
end

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

struct AttentionLayer{Q,K,V,O}
    wq::Q
    wk::K
    wv::V
    wo::O
    # n_heads::NH
    # head_size::HS
end

struct AttentionRMSNorm{W}
    weight::W
end

struct FFN{L, N}
    layers::L
    norm::N
end

struct FFNRMSNorm{W}
    weight::W
end

# FullModel => TransformerWeights
struct FullModel{T, N, O, TL}
    token_embedding_table::T
    rms_final_weight::N
    output_weight::O
    transformer_layers::TL
end

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


function make_new_layers(layers::Vector{<:TransformerLayerWeights})
    return Chain([make_new_layer(l) for l in layers]...)
end

function make_new_layer(l::TransformerLayerWeights)
    attention = AttentionLayer(
        l.wq,
        l.wk,
        l.wv,
        l.wo,
    )

    attention_rms = AttentionRMSNorm(
        l.rms_att_weight,
    )

    nn = Chain(
        Dense(l.w1'),
        Dense(l.w2'),
        Dense(l.w3'),
    )

    nn_norm = FFNRMSNorm(
        l.rms_ffn_weight,
    )
    ffn = FFN(nn, nn_norm)

    return TransformerLayer(attention, attention_rms, ffn)
end

function make_new_weights(t::TransformerWeights)
    layers = make_new_layers(t.layers)
    FullModel(
        t.token_embedding_table,
        t.rms_final_weight,
        t.output_weight,
        layers,
    )
end
# struct GGMLLlama
# end



function load_ggml_model(filename::AbstractString)
    header = nothing
    tokenizer = nothing
    tensor_dict = nothing

    open(filename) do file
        magic = read(file, UInt32)

        if magic != LLAMA_FILE_MAGIC_GGJT
            error("Only the GGJT_V2 file format is supported")
        end

        version = read(file, UInt32)

        if version != LLAMA_FILE_VERSION_GGJT_V2
            error("Only the GGJT_V2 file format is supported")
        end

        header = read_ggml_header(file)
        tokenizer = read_ggml_tokenizer(file, header.n_vocab)
        tensor_dict = read_ggml_tensor_dict(file)
    end

    # compute hidden dim
    n_ff_mult   = 2*(4*header.n_embd)÷3
    hidden_dim = ((n_ff_mult + header.n_mult - 1) ÷ header.n_mult) * header.n_mult

    weights = TransformerWeights(tensor_dict, header.n_layer)

    config = ModelConfig(;
        dim         = header.n_embd,
        hidden_dim,
        n_layers    = header.n_layer,
        n_heads     = header.n_head,
        n_kv_heads  = header.n_head,
        vocab_size  = header.n_vocab,
        seq_len     = 512, # FIXME: not sure what to put here
    )

    return LanguageModel(config, tokenizer, weights)
end
