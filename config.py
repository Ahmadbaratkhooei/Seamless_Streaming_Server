
SAMPLE_RATE = 16000
# TGT_LANG = "eng"
TGT_LANG = "eng"

def get_model_configs():
    """Returns the model configuration."""
    return {
        'monotonic_decoder_model_name': 'seamless_streaming_monotonic_decoder',
        'unity_model_name': 'seamless_streaming_unity',
        'sentencepiece_model': 'spm_256k_nllb100.model',
        'task': 's2st',
        'tgt_lang': TGT_LANG,
        'min_unit_chunk_size': 50,
        'decision_threshold': 0.6,
        'no_early_stop': True,
        'block_ngrams': True,
        'vocoder_name': 'vocoder_v2',
        'wav2vec_yaml': 'wav2vec.yaml',
        'min_starting_wait_w2vbert': 192,
        'config_yaml': 'cfg_fbank_u2t.yaml',
        'upstream_idx': 1,
        'detokenize_only': True,
        'device': 'cuda:0',
        'dtype': 'fp16',
        'max_len_a': 0,
        'max_len_b': 1000,
    }