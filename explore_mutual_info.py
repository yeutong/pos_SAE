# %%
import einops
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name, tokenize_and_concatenate

from sae_lens.toolkit.pretrained_saes import get_gpt2_res_jb_saes
from sae_lens.analysis.neuronpedia_integration import get_neuronpedia_quick_list

from datasets import load_dataset
from jaxtyping import Int, Float

import plotly.express as px
import plotly.graph_objects as go

torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# %%
layer = 0
model = HookedTransformer.from_pretrained("gpt2-small")
hook_name = get_act_name("resid_pre", layer)
saes, sparsities = get_gpt2_res_jb_saes(hook_name)
sae = saes[hook_name].to(device)

# %%
n_ctx = 256
n_token = 400
sample_tokens = torch.randperm(model.cfg.d_vocab)[:n_token]  # without replacement
emb = model.W_E[sample_tokens].unsqueeze(dim=1)
pos = model.W_pos[:n_ctx].unsqueeze(dim=0)
resid_pre_emb_pos: Float[Tensor, "tok pos d_model"] = emb + pos
feature_acts: Float[Tensor, "tok pos n_features"] = sae(resid_pre_emb_pos)[1]
# feature_acts: Float[Tensor, "tok pos n_features"] = sae._encode_with_hidden_pre(resid_pre_emb_pos)[1] # use this if you want pre-relu activations

# %%
# naive method: select the first sae.cfg.context_size position, take the mean over the token dimension, then take the max over the position dimension

naive_method_score = feature_acts[: sae.cfg.context_size].mean(dim=0).max(dim=0).values
top_k_features = 50
pos_features = naive_method_score.topk(top_k_features).indices


def plot_mean_act_by_pos(feature_acts, pos_features, plot_until_pos=128):
    activation_mean_by_pos = feature_acts.mean(dim=0)[:, pos_features]
    fig = go.Figure()
    for i, feature in enumerate(pos_features):
        fig.add_trace(
            go.Scatter(
                y=activation_mean_by_pos[:plot_until_pos, i].cpu(),
                mode="lines",
                name=f"feature {feature}",
            )
        )
    fig.update_layout(
        title="Mean Feature Activation by Position",
        xaxis_title="Position",
        yaxis_title="Feature Activation",
        width=800,
    )
    fig.show()


plot_mean_act_by_pos(feature_acts, pos_features, 128)
plot_mean_act_by_pos(feature_acts, pos_features, 256)


# %%
# Calcualate Mutual Information between feature act and position

feature_freq_pos: Float[Tensor, "pos n_features"] = (feature_acts > 0).sum(
    dim=0
) / n_token
feature_freq: Float[Tensor, "n_features"] = feature_freq_pos.mean(dim=0)

p: Float[Tensor, "pos n_features"] = feature_freq_pos
q: Float[Tensor, "1 n_features"] = feature_freq.unsqueeze(0)

# Compute mutual information for each position and feature
mutual_info_pos_detail: Float[Tensor, "pos n_features"] = p * torch.log(p / q) + (
    1 - p
) * torch.log((1 - p) / (1 - q))

# Aggregate mutual information across positions
mutual_info_pos: Float[Tensor, "n_features"] = mutual_info_pos_detail.mean(dim=0)
mutual_info_pos = mutual_info_pos.nan_to_num(0)
# px.histogram(mutual_info_pos.cpu(), log_y=True, title="Mutual Information Histogram")

pos_features = mutual_info_pos.topk(50).indices
activation_mean_by_pos = feature_acts.mean(dim=0)[:, pos_features]
fig = go.Figure()
for i, pos in enumerate(pos_features):
    fig.add_trace(
        go.Scatter(
            y=activation_mean_by_pos[:20, i].cpu(), mode="lines", name=f"pos {pos}"
        )
    )
fig.update_layout(title="Mean Activation by Position")
fig.show()

px.scatter(
    x=naive_method_score.cpu(),
    y=mutual_info_pos.cpu(),
    title="Naive Method Score vs Mutual Information",
    hover_data=[range(sae.cfg.d_sae)],
)
