"""
12/05/2024
"""

# %%
import einops
import torch
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name

from sae_lens.toolkit.pretrained_saes import get_gpt2_res_jb_saes
from sae_lens.analysis.neuronpedia_integration import get_neuronpedia_quick_list

from jaxtyping import Int, Float
from torch import Tensor
import plotly.express as px

torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"

from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name, tokenize_and_concatenate

from datasets import load_dataset

import plotly.graph_objects as go

# %%
layer = 0
model = HookedTransformer.from_pretrained("gpt2-small")
hook_name = get_act_name("resid_pre", layer)
saes, sparsities = get_gpt2_res_jb_saes(hook_name)
sae = saes[hook_name].to(device)
# %%

# shape: d_vocab, n_ctx, d_model
# pass in sae, get d_vocab, n_ctx, n_features
# mean/sum over d_vocab, get top k features for each position

# %%

# COPIED FROM YESTERDAY

n_ctx = 256
n_token = 400
torch.manual_seed(0)
sample_tokens = torch.randint(0, model.cfg.d_vocab, (n_token,)).to(device)
emb = model.W_E[sample_tokens].unsqueeze(dim=1)
pos = model.W_pos[:n_ctx].unsqueeze(dim=0)
resid_pre_emb_pos = emb + pos
resid_pre_flatten = einops.rearrange(
    resid_pre_emb_pos, "d_vocab n_ctx d_model -> (d_vocab n_ctx) d_model"
)
feature_acts = sae(resid_pre_flatten)[1]
feature_acts_unflattened = einops.rearrange(
    feature_acts, "(d_vocab n_ctx) d_sae -> d_vocab n_ctx d_sae", n_ctx=n_ctx
)
feature_acts_meaned = feature_acts_unflattened.mean(dim=0)
feature_acts_count = (feature_acts_unflattened > 0).sum(dim=0)
top_k = torch.topk(feature_acts_count, k=10, dim=-1).indices

# %%

# interesting_features: 2, 6

for focus_feature in range(0, 10):
    focus_feature_acts = feature_acts_unflattened[..., focus_feature]

    px.imshow(
        focus_feature_acts[:].cpu().numpy(),
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBU",
        labels={"x": "pos", "y": "tok"},
        title=f"Feature {focus_feature}",
    ).show()

# px.line(focus_feature_acts[314].cpu().numpy()).show()

# %%

# pick feature 41.

focus_feature = 56
focus_feature_acts = feature_acts_unflattened[..., focus_feature]
accidental_token_indices = torch.topk(focus_feature_acts.mean(dim=-1), k=3).indices
accidental_tokens = sample_tokens[accidental_token_indices]

for accidental_token in accidental_tokens:
    str_token = model.to_single_str_token(accidental_token.item())
    print(str_token)

get_neuronpedia_quick_list([focus_feature], layer)

# %%
print("leep" * 256)

# %%
focus_feature = 56
focus_feature_acts = feature_acts_unflattened[..., focus_feature]
accidental_token_indices = torch.topk(focus_feature_acts.mean(dim=-1), k=5).indices
accidental_tokens = sample_tokens[accidental_token_indices]

fig = go.Figure()
for accidental_token in accidental_token_indices:
    str_token = model.to_single_str_token(accidental_token.item())
    fig.add_trace(
        go.Scatter(
            y=focus_feature_acts[accidental_token.item(), :].cpu().numpy(),
            mode="lines",
            name=str_token,
        )
    )

# add a vertical line at 128
fig.add_shape(
    type="line", x0=128, x1=128, y0=0, y1=0.1, line=dict(color="black", width=1)
)
fig.show()

# %%
sae.cfg.context_size
