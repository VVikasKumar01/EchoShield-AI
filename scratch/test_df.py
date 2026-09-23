import torch
import torchaudio
from df.enhance import init_df, df_features, get_device
from df.io import load_audio

model, df_state, suffix, epoch = init_df()
print("Model type:", type(model))
print("Model pad_feat:", getattr(model, "pad_feat", None))
audio, _ = load_audio("assets/noisy_snr0.wav", sr=48000)
nb_df = getattr(model, "nb_df", getattr(model, "df_bins", 0))
spec, erb_feat, spec_feat = df_features(audio, df_state, nb_df, device=get_device())
print("spec shape:", spec.shape)
print("erb_feat shape:", erb_feat.shape)
print("spec_feat shape:", spec_feat.shape)
try:
    print("pad_feat padding:", model.pad_feat.padding)
    out_erb = model.pad_feat(erb_feat)
    print("out_erb shape:", out_erb.shape)
except Exception as e:
    import traceback
    traceback.print_exc()
