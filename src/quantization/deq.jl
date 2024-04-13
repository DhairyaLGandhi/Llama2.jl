function dequantize(model::LanguageModel{T,W}) where {T,W}
    weights = dequantize(model.weights)
    return LanguageModel(model.config,
                         model.tokenizer,
                         weights)
end
function dequantize(weights::TransformerWeights)
    return TransformerWeights(dequantize(weights.token_embedding_table),
                              map(dequantize, weights.layers),
                              weights.rms_final_weight,
                              dequantize(weights.output_weight))
end
function dequantize(weights::TransformerLayerWeights)
    return TransformerLayerWeights(weights.rms_att_weight,
                                   weights.rms_ffn_weight,
                                   [dequantize(getfield(weights, field)) for field in fieldnames(TransformerLayerWeights)[3:end]]...)
end
dequantize(M::Matrix) =
    reshape(dequantize(reshape(M, length(M))), size(M)[1]*QK_K, size(M)[2])