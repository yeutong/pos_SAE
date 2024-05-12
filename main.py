"""
10/05/2024
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
n_ctx = 256
n_token = 400
torch.manual_seed(0)
sample_tokens = torch.randint(0, model.cfg.d_vocab, (n_token,))
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
focus_pos = 1
focus_pos_features = top_k[focus_pos]
get_neuronpedia_quick_list(focus_pos_features.tolist(), layer)


# %%
str_toks = model.to_str_tokens(
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
)

for i, str_tok in enumerate(str_toks):
    print(f"{i}: {repr(str_tok)}")


# %%
k = 1000
first_k_pos_embeds = model.W_pos[:k]
first_k_pos_embeds_normed = first_k_pos_embeds / first_k_pos_embeds.norm(
    dim=-1, keepdim=True
)
cosine_sims = einops.einsum(
    first_k_pos_embeds_normed,
    first_k_pos_embeds_normed,
    "pos1 d_model, pos2 d_model -> pos1 pos2",
)
px.imshow(cosine_sims.cpu().numpy()).show()


# %%
def show_cosine_sims(
    v1: Float[Tensor, "batch1 d"],
    v2: Float[Tensor, "batch2 d"],
    label1: str,
    label2: str,
    **kwargs,
) -> None:
    v1_norm = v1 / v1.norm(dim=-1, keepdim=True)
    v2_norm = v2 / v2.norm(dim=-1, keepdim=True)
    cosine_sims = einops.einsum(v1_norm, v2_norm, "batch1 d, batch2 d -> batch1 batch2")
    px.imshow(
        cosine_sims.cpu().numpy(), labels={"x": label2, "y": label1}, **kwargs
    ).show()


# %%

# check the features are correlated with W_pos
pos = 5
top_k: Int[Tensor, "n_ctx k"]
gt_pos_features: Float[Tensor, "n_ctx d_model"] = model.W_pos[:20]


enc_top_k = sae.W_enc[:, top_k]
enc_top_k = einops.rearrange(enc_top_k, "d_model n_ctx k -> n_ctx d_model k")
focus_enc_top_k = enc_top_k[pos].T
show_cosine_sims(
    focus_enc_top_k, gt_pos_features, f"sae feature for pos {pos}", "W_pos", title="Enc"
)


dec_top_k = sae.W_dec.T[:, top_k]
dec_top_k = einops.rearrange(dec_top_k, "d_model n_ctx k -> n_ctx d_model k")
focus_dec_top_k = dec_top_k[pos].T
show_cosine_sims(
    focus_dec_top_k, gt_pos_features, f"sae feature for pos {pos}", "W_pos", title="Dec"
)

# %%
gt_tok_features: Float[Tensor, "tok d_model"] = model.W_E[24000:24100]
show_cosine_sims(
    focus_dec_top_k,
    gt_tok_features,
    f"sae feature for pos {pos}",
    "W_E[:100]",
    title="Dec",
)


# %%
max_length = 256
openwebtext = load_dataset("stas/openwebtext-10k", split="train")
dataset = tokenize_and_concatenate(openwebtext, model.tokenizer, max_length=max_length)
data_loader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=True)
tokens = next(iter(data_loader))["tokens"]

# %%
logits, cache = model.run_with_cache(tokens, names_filter=["blocks.0.hook_resid_pre"])
resid_pre = cache[sae.cfg.hook_point]
resid_pre_flattened = einops.rearrange(
    resid_pre, "batch seq d_model -> (batch seq) d_model"
)
sae_out = sae(resid_pre_flattened)[0]
sae_out_expanded = einops.rearrange(
    sae_out, "(batch seq) d_model -> batch seq d_model", batch=resid_pre.shape[0]
)
re = (sae_out_expanded - resid_pre).norm(dim=(0, -1))
px.line(re.cpu().numpy()).show()

# try and fix this
# %%
correction: Float[Tensor, "batch seq d_model"] = torch.zeros_like(sae_out_expanded)
delta = model.W_pos[
    sae.cfg.context_size : sae.cfg.context_size + (max_length - sae.cfg.context_size)
]
correction[:, sae.cfg.context_size :] = delta
sae_out_expanded_fixed = sae_out_expanded + correction

# subtract the component of sae_out_expanded in the correction direction before doing this operation
# sae_out_expanded_fixed_norm = sae_out_expanded_fixed_norm / sae_out_expanded_fixed_norm.norm(dim=-1, keepdim=True)

re_fixed = (sae_out_expanded_fixed - resid_pre).norm(dim=(0, -1))
px.line(re_fixed.cpu().numpy()).show()


"""
11/05/2024
"""


# %%

focus_pos = 127
features = top_k[focus_pos].tolist()
tokens = next(iter(data_loader))["tokens"]
logits, cache = model.run_with_cache(tokens, names_filter=["blocks.0.hook_resid_pre"])
resid_pre = cache[sae.cfg.hook_point]
_, hidden_pre = sae._encode_with_hidden_pre(resid_pre)
hidden_pre_std = hidden_pre.std(dim=0)
hidden_pre_mean = hidden_pre.mean(dim=0)
hidden_pre_mean_features = hidden_pre_mean[:, features]
hidden_pre_std_features = hidden_pre_std[:, features]

fig = go.Figure()
for i, feature in enumerate(features):
    fig.add_trace(
        go.Line(
            y=hidden_pre_mean_features[:, i].cpu().numpy(),
            name=f"feature {feature}",
            error_y=dict(
                type="data", array=hidden_pre_std_features[:, i].cpu().numpy()
            ),
        )
    )
fig.show()

# %%
# trying to asign a score to each (pos, feature) pair, and take the top k total features
# scores: [n_ctx d_sae]
_, hidden_pre = sae._encode_with_hidden_pre(resid_pre_emb_pos)
feature_acts_meaned = hidden_pre.mean(dim=0)
feature_acts_std = hidden_pre.std(dim=0)
t_scores = feature_acts_meaned / feature_acts_std

# %%
max_t_score_per_feature = t_scores.max(dim=0).values
px.histogram(max_t_score_per_feature.cpu().numpy(), log_y=True, marginal="box").show()
pos_features = max_t_score_per_feature > 0
num_pos_features = pos_features.sum()
print(f"num_pos_features: {num_pos_features}")
pos_feature_indices = torch.where(pos_features)[0].tolist()
# %%
hidden_pre_mean_features = hidden_pre_mean[:, pos_features]
hidden_pre_std_features = hidden_pre_std[:, pos_features]
fig = go.Figure()
for i in range(num_pos_features):
    fig.add_trace(
        go.Line(
            y=hidden_pre_mean_features[:, i].cpu().numpy(),
            name=f"feature {pos_feature_indices[i]}",
        )
    )  # , error_y=dict(type='data', array=hidden_pre_std_features[:, i].cpu().numpy())))
fig.show()

# %%
# ablate the positional features
tokens = next(iter(data_loader))["tokens"]
logits, cache = model.run_with_cache(tokens, names_filter=["blocks.0.hook_resid_pre"])
resid_pre = cache[sae.cfg.hook_point]
sae_out_orig, feature_acts = sae(resid_pre)[:2]

feature_acts[:, :, pos_features] = 0
sae_out_ablated = sae.decode(feature_acts)

correction: Float[Tensor, "batch seq d_model"] = torch.zeros_like(sae_out_expanded)
delta = model.W_pos[:max_length]
# mean_pos = model.W_pos[:128].mean(dim=0)
sae_out_fixed = sae_out_ablated + delta

mean = (sae_out_orig - sae_out_fixed).mean(dim=(0))[:128].mean(dim=0)

sae_out_fixed_plus_mean = sae_out_fixed + mean


re_fixed_plus_mean = (sae_out_fixed_plus_mean - resid_pre).norm(dim=(0, -1))
re_fixed = (sae_out_fixed - resid_pre).norm(dim=(0, -1))
re_orig = (sae_out_orig - resid_pre).norm(dim=(0, -1))
re_ablated = (sae_out_ablated - resid_pre).norm(dim=(0, -1))
fig = go.Figure()
fig.add_trace(go.Line(y=re_fixed.cpu().numpy(), name="fixed"))
fig.add_trace(go.Line(y=re_orig.cpu().numpy(), name="orig"))
fig.add_trace(go.Line(y=re_ablated.cpu().numpy(), name="ablated"))
fig.add_trace(go.Line(y=re_fixed_plus_mean.cpu().numpy(), name="fixed_plus_mean"))
fig.show()


# re = (sae_out_expanded - resid_pre).norm(dim=(0,-1))
# px.line(re.cpu().numpy()).show()

# %%
focus_tok = 15
tok = sample_tokens[focus_tok].item()
print(model.to_single_str_token(tok))
focus_features = feature_acts_unflattened[focus_tok]
focus_features_mean = focus_features[:128].mean(dim=0)
top_k_features_for_token = torch.topk(focus_features_mean, k=10).indices
get_neuronpedia_quick_list(top_k_features_for_token.tolist(), layer)

# %%
features_to_plot = focus_features[:, top_k_features_for_token]

fig = go.Figure()
for i in range(10):
    fig.add_trace(
        go.Line(
            y=features_to_plot[:, i].cpu().numpy(),
            name=f"feature {top_k_features_for_token[i]}",
        )
    )
fig.show()


# %%
"""
ablate everything that is not a flat line on a plot of the form feature act vs pos for a fixed token
"""
# calc inverse token_score
n_ctx = 128
batch_i = 0
batch_size = 200

min_stds = torch.ones(sae.cfg.d_sae, device=device) * float("inf")
min_token_ids = torch.zeros(sae.cfg.d_sae, dtype=torch.int, device=device)

for batch_i in range(model.cfg.d_vocab // batch_size + 1):
    start_token = batch_i * batch_size
    end_token = (batch_i + 1) * batch_size
    print(f"batch_i: {batch_i}, start_token: {start_token}, end_token: {end_token}")

    emb = model.W_E[start_token:end_token].unsqueeze(dim=1)
    pos = model.W_pos[:n_ctx].unsqueeze(dim=0)
    resid_pre_emb_pos = emb + pos
    hidden_pre: Float[Tensor, "batch pos n_features"] = sae._encode_with_hidden_pre(
        resid_pre_emb_pos
    )[1]
    # for each features, get std along pos dimention
    stds: Float[Tensor, "batch n_features"] = hidden_pre.std(dim=1)

    tmp_stds, tmp_token_id = stds.min(dim=0)

    min_token_ids = torch.where(
        tmp_stds < min_stds, start_token + tmp_token_id, min_token_ids
    )
    min_stds = torch.where(tmp_stds < min_stds, tmp_stds, min_stds)

# %%
# calc inverse pos score
n_ctx = 128

min_stds_pos = torch.ones(sae.cfg.d_sae, device=device) * float("inf")
min_token_ids_pos = torch.zeros(sae.cfg.d_sae, dtype=torch.int, device=device)

for pos in range(n_ctx):
    print(pos)

    emb = model.W_E.unsqueeze(dim=1)
    pos = model.W_pos[[pos]].unsqueeze(dim=0)
    resid_pre_emb_pos = emb + pos
    hidden_pre: Float[Tensor, "batch pos n_features"] = sae._encode_with_hidden_pre(
        resid_pre_emb_pos
    )[1]
    # for each features, get std along pos dimention
    stds: Float[Tensor, "batch n_features"] = hidden_pre.std(dim=0)

    tmp_stds, tmp_token_id = stds.min(dim=0)

    min_token_ids_pos = torch.where(
        tmp_stds < min_stds_pos, tmp_token_id, min_token_ids_pos
    )
    min_stds_pos = torch.where(tmp_stds < min_stds_pos, tmp_stds, min_stds_pos)


# %%
px.scatter(x=min_stds.cpu().numpy(), y=min_stds_pos.cpu().numpy()).show()
# %%
# px.histogram(min_stds.cpu().numpy(), log_y=True, marginal="box").show()
# px.histogram(min_token_ids.cpu().numpy(), log_y=True, marginal="box").show()
# %%
thres = 0.01
token_features = min_stds < thres
print(f"Token features: {token_features.sum()}")
print(f"Ablated features: {(~token_features).sum()}")
# %%
# keep token features, ablate the rest
tokens = next(iter(data_loader))["tokens"]
logits, cache = model.run_with_cache(tokens, names_filter=["blocks.0.hook_resid_pre"])
resid_pre = cache[sae.cfg.hook_point]
sae_out_orig, feature_acts = sae(resid_pre)[:2]

feature_acts[:, :, ~token_features] = 0
sae_out_ablated = sae.decode(feature_acts)
delta = model.W_pos[:max_length]
# mean_pos = model.W_pos[:128].mean(dim=0)
sae_out_fixed = sae_out_ablated + delta

mean = (sae_out_orig - sae_out_fixed).mean(dim=(0))[:128].mean(dim=0)

sae_out_fixed_plus_mean = sae_out_fixed + mean


re_fixed_plus_mean = (sae_out_fixed_plus_mean - resid_pre).norm(dim=(0, -1))
re_fixed = (sae_out_fixed - resid_pre).norm(dim=(0, -1))
re_orig = (sae_out_orig - resid_pre).norm(dim=(0, -1))
re_ablated = (sae_out_ablated - resid_pre).norm(dim=(0, -1))
fig = go.Figure()
fig.add_trace(go.Line(y=re_fixed.cpu().numpy(), name="fixed"))
fig.add_trace(go.Line(y=re_orig.cpu().numpy(), name="orig"))
fig.add_trace(go.Line(y=re_ablated.cpu().numpy(), name="ablated"))
fig.add_trace(go.Line(y=re_fixed_plus_mean.cpu().numpy(), name="fixed_plus_mean"))
# add title
fig.update_layout(
    title=f"Ablated features: {(~token_features).sum()}, std thres: {thres}"
)
fig.show()
# %%
