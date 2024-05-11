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
import pandas as pd

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
n_token = 100
torch.manual_seed(0)
sample_tokens = torch.randint(0, model.cfg.d_vocab, (n_token,))
emb = model.W_E[sample_tokens].unsqueeze(dim=1)
pos = model.W_pos[:n_ctx].unsqueeze(dim=0)
resid_pre = emb + pos
resid_pre_flatten = einops.rearrange(
    resid_pre, "d_vocab n_ctx d_model -> (d_vocab n_ctx) d_model"
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
gt_tok_features: Float[Tensor, "tok d_moel"] = model.W_E[24000:24100]
show_cosine_sims(
    focus_dec_top_k,
    gt_tok_features,
    f"sae feature for pos {pos}",
    "W_E[:100]",
    title="Dec",
)


# %%
max_length = 200
openwebtext = load_dataset("stas/openwebtext-10k", split="train")
dataset = tokenize_and_concatenate(openwebtext, model.tokenizer, max_length=max_length)
data_loader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=True)
tokens = next(iter(data_loader))["tokens"]

# %%

logits, cache = model.run_with_cache(tokens, names_filter=[sae.cfg.hook_point])
resid_pre = cache[sae.cfg.hook_point]
# resid_pre_flattened = einops.rearrange(resid_pre, "batch seq d_model -> (batch seq) d_model")
# sae_out = sae(resid_pre_flattened).sae_out
# sae_out_expanded = einops.rearrange(sae_out, "(batch seq) d_model -> batch seq d_model", batch=resid_pre.shape[0])
sae_out_expanded: Float[Tensor, "batch seq d_model"] = sae(resid_pre).sae_out
re = (sae_out_expanded - resid_pre).norm(dim=(0, -1))
px.line(re.cpu().numpy()).show()

# try and fix this
# %%
correction: Float[Tensor, "batch seq d_model"] = torch.zeros_like(sae_out_expanded)
delta = model.W_pos[sae.cfg.context_size : max_length]
correction[:, sae.cfg.context_size :] = delta
sae_out_expanded_fixed = sae_out_expanded + correction
re_fixed = (sae_out_expanded_fixed - resid_pre).norm(dim=(0, -1))
px.line(re_fixed.cpu().numpy()).show()


# subtract the component of sae_out_expanded in the correction direction before doing this operation
def get_A_projection_on_B(A, B):
    magnitude = (A * B).sum(dim=-1) / B.norm(dim=-1)
    direction = B / B.norm(dim=-1, keepdim=True)
    return magnitude.unsqueeze(dim=-1) * direction


sae_out_expanded_fixed_remove_component = (
    sae_out_expanded + correction - get_A_projection_on_B(correction, sae_out_expanded)
)
re_fixed_remove_component = (sae_out_expanded_fixed_remove_component - resid_pre).norm(
    dim=(0, -1)
)
px.line(re_fixed.cpu().numpy()).show()


# %%
df = pd.DataFrame(
    {
        "Original": re.cpu().numpy(),
        "Add W_pos ": re_fixed.cpu().numpy(),
        "Add othogonal W_pos component": re_fixed_remove_component.cpu().numpy(),
    }
)

# Using Plotly Express to plot both lines on the same graph
fig = px.line(df, title="RE for sae out")
fig.show()

# %%
# inspect W_E and W_pos max cos sim
W_E_norm = model.W_E / model.W_E.norm(dim=-1, keepdim=True)
W_pos_norm = model.W_pos / model.W_pos.norm(dim=-1, keepdim=True)
WE_WPos_cos_sim = einops.einsum(
    W_E_norm, W_pos_norm, "tok d_model, pos d_model -> tok pos"
)

# for each pos, get max
max_cos_sim = WE_WPos_cos_sim.max(dim=0).values

px.line(
    max_cos_sim.cpu().numpy(),
    title="Max cos sim between W_pos and W_E for each pos",
    labels={"index": "pos", "value": "max cos sim"},
).show()


# %%
# get features whose W_dec is correlated with W_pos[:128]

W_dec: Float[Tensor, "n_features d_model"] = sae.W_dec
W_pos_ctx: Float[Tensor, "n_ctx d_model"] = model.W_pos[: sae.cfg.context_size]

W_dec_norm = W_dec / W_dec.norm(dim=-1, keepdim=True)
W_pos_ctx_norm = W_pos_ctx / W_pos_ctx.norm(dim=-1, keepdim=True)

WDec_WPos_ctx_cos_sim = einops.einsum(
    W_dec_norm, W_pos_ctx_norm, "n_features d_model, n_ctx d_model -> n_features n_ctx"
)

# %%
# for each feature, get max cos sim
max_cos_sim = WDec_WPos_ctx_cos_sim.max(dim=1).values

px.histogram(
    max_cos_sim.cpu().numpy(),
    title="Max cos sim between W_dec and W_pos[:128] for each feature",
    labels={"value": "max cos sim"},
    log_y=True,
).show()


# %%
min_cos_sim = WDec_WPos_ctx_cos_sim.min(dim=1).values

px.histogram(
    min_cos_sim.cpu().numpy(),
    title="Min cos sim between W_dec and W_pos[:128] for each feature",
    labels={"value": "min cos sim"},
    log_y=True,
).show()


# %%
# get features whose max cos sim with W_E (max_cos_sim) > 0.5
target_features = max_cos_sim > 0.5
print(target_features.sum())

# %%
# calc re loss

# %%
# NOTES
# - do statistics, look at std of the feature act if it is a positional feature
# - position 0 slightly fucked as BOS token is always 0 in training data
