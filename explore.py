# %%
import torch
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name, tokenize_and_concatenate

from datasets import load_dataset
from sae_lens.toolkit.pretrained_saes import get_gpt2_res_jb_saes

import pandas as pd
import plotly.express as px

torch.set_grad_enabled(False)

# %%
model = HookedTransformer.from_pretrained("gpt2-small")

hook_name = get_act_name("resid_pre", 8)
saes, sparsities = get_gpt2_res_jb_saes(hook_name)
sae = saes[hook_name]
# %%
openwebtext = load_dataset("stas/openwebtext-10k", split='train')
dataset = tokenize_and_concatenate(openwebtext, model.tokenizer, max_length=30)
data_loader = DataLoader(dataset, batch_size=16, shuffle=False, drop_last=True)
tokens = next(iter(data_loader))['tokens']

del openwebtext
del dataset

# %%
# get resid pre 8 cache from random prompt
_, cache = model.run_with_cache(tokens, names_filter=hook_name)

resid = cache[hook_name]

# %%
sae.to(resid.device)
sae_out, feature_acts, loss, mse_loss, l1_loss, ghost_grad_loss = sae(resid)

# feature_acts with shape: batch, pos, n_features

# %%
# fix pos 10, find features that always fire on pos 10
pos = 10

# see
px.histogram(feature_acts[:, pos, :].mean(dim=0).cpu(), log_y=True).show()


# %%
act_pos_mean = feature_acts[:, pos, :].mean(dim=0) # shape: n_features
act_mean = feature_acts.mean(dim=(0, 1)) # shape: n_features


px.histogram((act_pos_mean - act_mean).cpu(), log_y=True)
# %%
# 

((act_pos_mean - act_mean) > 1).sum()
# %%
target_features = (act_pos_mean - act_mean) > 1
target_features_act = feature_acts[:, :, target_features]

# %%
target_features_act.shape
# %%
# px.scatter(target_features_act[:, :, 0].cpu())

df = pd.DataFrame(target_features_act[:, :, 9].cpu().numpy())
df_melted = df.melt(var_name='x', value_name='y')
fig = px.scatter(df_melted, x='x', y='y')
fig.show()

# %%
