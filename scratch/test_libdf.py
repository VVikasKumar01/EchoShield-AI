import torch
import numpy as np
from df.enhance import init_df, get_norm_alpha
from df.io import load_audio
from libdf import DF, erb, erb_norm, unit_norm

model, df_state, suffix, epoch = init_df()
audio, _ = load_audio("assets/noisy_snr0.wav", sr=48000)
spec = df_state.analysis(audio.numpy())
print("spec shape:", spec.shape, "dtype:", spec.dtype)
erb_fb = df_state.erb_widths()
print("erb_fb shape / type:", type(erb_fb), getattr(erb_fb, "shape", None), len(erb_fb))
e = erb(spec, erb_fb)
print("erb(spec, erb_fb) shape:", type(e), getattr(e, "shape", None))
a = get_norm_alpha(False)
print("a:", a)
en = erb_norm(e, a)
print("erb_norm shape:", type(en), getattr(en, "shape", None))
