"""Real model implementations shared by repaired GDM41–44.

All backbones are frozen in this screening repair. Adapters here are explicitly
FEATURE adapters, not transformer fine-tuning. No architecture gets fabricated metrics.
"""
from __future__ import annotations
import hashlib
import re
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .contracts import SPECIAL, options, option_text, sha

REGISTRY = {'distilbert': 'distilbert/distilbert-base-uncased',
            'modernbert': 'answerdotai/ModernBERT-base',
            'mmbert': 'jhu-clsp/mmBERT-base', 'neobert': 'chandar-lab/NeoBERT',
            'qwen': 'xthor/Qwen3-Embedding-0.6B-GraphQL'}


def install_neobert_cpu_xformers_shim():
    """Install the exact eager PyTorch ops NeoBERT needs on CPU.

    NeoBERT's pinned remote code imports ``xformers.ops.SwiGLU`` and
    ``memory_efficient_attention`` unconditionally. Published xformers wheels are
    CUDA-oriented and fail to import on GitHub's CPU runners before xformers can
    reach its own eager fallback. This shim preserves the same parameter names
    and eager equations used by xformers while avoiding any CUDA extension.
    """
    import sys
    import types

    if 'xformers.ops' in sys.modules:
        return

    class SwiGLU(nn.Module):
        def __init__(self, in_features, hidden_features, out_features=None, bias=True, **_):
            super().__init__()
            out_features = out_features or in_features
            hidden_features = hidden_features or in_features
            self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
            self.w3 = nn.Linear(hidden_features, out_features, bias=bias)
            self.hidden_features = hidden_features
            self.out_features = out_features
            self.in_features = in_features

        def forward(self, x):
            x1, x2 = self.w12(x).chunk(2, dim=-1)
            return self.w3(F.silu(x1) * x2)

    def memory_efficient_attention(query, key, value, attn_bias=None, p=0.0, scale=None, **_):
        # xformers accepts [B, M, H, K]; PyTorch SDPA accepts [B, H, M, K].
        result = F.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
            attn_mask=attn_bias, dropout_p=p, scale=scale,
        )
        return result.transpose(1, 2)

    ops = types.ModuleType('xformers.ops')
    ops.SwiGLU = SwiGLU
    ops.memory_efficient_attention = memory_efficient_attention
    package = types.ModuleType('xformers')
    package.ops = ops
    sys.modules['xformers'] = package
    sys.modules['xformers.ops'] = ops


class Encoder:
    def __init__(self, name, revision=None):
        self.name = name; self.calls = 0; self.texts = 0; self.cache = {}
        if name == 'hash':
            self.dim = 128
            self.meta = {'model_id': 'hash-control', 'revision': 'sha256-word-bigram-v1',
                         'parameters': 0, 'role': 'unit-test/control only'}
            return
        from huggingface_hub import HfApi
        if name == 'neobert':
            install_neobert_cpu_xformers_shim()
        from transformers import AutoModel, AutoTokenizer
        model_id = REGISTRY[name]
        revision = revision or HfApi().model_info(model_id).sha
        if not revision or len(revision) != 40:
            raise ValueError('Model revision must resolve to a commit hash')
        remote = name == 'neobert'
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=remote)
        kwargs = {'revision': revision, 'trust_remote_code': remote, 'torch_dtype': torch.float32}
        if not remote:
            kwargs['attn_implementation'] = 'eager'
        self.model = AutoModel.from_pretrained(model_id, **kwargs).cpu().eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.dim = self.model.config.hidden_size
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.meta = {'model_id': model_id, 'revision': revision,
                     'parameters': sum(p.numel() for p in self.model.parameters()),
                     'pooling': 'last-token' if name == 'qwen' else ('cls' if name == 'neobert' else 'mean'),
                     'dtype': 'float32', 'device': 'cpu', 'max_tokens': 256,
                     'frozen': True, 'remote_code_pinned': remote,
                     'cpu_compatibility': 'pytorch-eager-xformers-equivalent' if name == 'neobert' else None}

    def encode(self, texts, cached=True, batch_size=8):
        texts = list(texts)
        missing = list(dict.fromkeys(t for t in texts if not cached or t not in self.cache))
        fresh = {}
        for start in range(0, len(missing), batch_size):
            group = missing[start:start + batch_size]
            self.calls += 1; self.texts += len(group)
            if self.name == 'hash':
                mat = torch.zeros(len(group), self.dim)
                for i, text in enumerate(group):
                    words = re.findall(r'[A-Za-z0-9_]+', text.lower())
                    tokens = words + [a + '/' + b for a, b in zip(words, words[1:])]
                    for token in tokens:
                        code = hashlib.sha256(token.encode()).digest()
                        mat[i, int.from_bytes(code[:4], 'little') % self.dim] += 1 if code[4] % 2 else -1
            else:
                encoded = self.tokenizer(group, padding=True, truncation=False, return_tensors='pt')
                if encoded['input_ids'].shape[1] > 256:
                    raise ValueError('Refusing silent input truncation')
                with torch.no_grad():
                    hidden = self.model(**encoded).last_hidden_state
                    mask = encoded['attention_mask']
                    if self.name == 'qwen':
                        positions = torch.arange(mask.shape[1]).expand_as(mask)
                        last = positions.masked_fill(mask == 0, -1).max(1).values
                        mat = hidden[torch.arange(len(group)), last]
                    elif self.name == 'neobert':
                        mat = hidden[:, 0]
                    else:
                        mat = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
            mat = F.normalize(mat.float(), dim=-1).detach()
            if not torch.isfinite(mat).all():
                raise FloatingPointError('Nonfinite embedding')
            for text, vector in zip(group, mat):
                fresh[text] = vector
                if cached:
                    self.cache[text] = vector
        return torch.stack([fresh[t] if t in fresh else self.cache[t] for t in texts])

    def query(self, request, cached=True):
        if self.name == 'qwen':
            request = 'Instruct: Retrieve GraphQL field paths satisfying the request.\nQuery: ' + request
        return self.encode([request], cached=cached)[0]


def candidate_units(item, tree=False, typed=True):
    opts = options(item)
    if not tree:
        return [{'id': o['id'], 'path': o['path'], 'members': [o['id']],
                 'text': option_text(o, typed), 'parent': ''} for o in opts]
    units = {}
    for o in opts:
        if o['id'] in SPECIAL:
            units[o['id']] = {'id': o['id'], 'path': [], 'members': [o['id']], 'text': SPECIAL[o['id']], 'parent': ''}
            continue
        for length in range(1, len(o['path']) + 1):
            prefix = o['path'][:length]
            key = '.'.join(prefix)
            if key not in units:
                units[key] = {'id': key, 'path': prefix, 'members': [], 'parent': '.'.join(prefix[:-1]),
                              'text': 'Current graph scope ' + ('.'.join(prefix[:-1]) or 'Query') +
                                      '; select field ' + prefix[-1] + '; path ' + key}
            units[key]['members'].append(o['id'])
    # Public catalog descriptions only: no request targets in these state descriptors.
    for u in units.values():
        if u['id'] not in SPECIAL:
            endings = [o for o in opts if o['id'] in u['members']]
            u['text'] += '; reaches ' + ' | '.join(o['gloss'] for o in endings)
            if typed:
                u['text'] += '; possible terminal signatures ' + ', '.join(sorted({o['types'][-1] for o in endings}))
    return list(units.values())


class Features:
    def __init__(self, encoder, config, retriever=None):
        self.encoder = encoder; self.config = config; self.retriever = retriever
        self.dim = encoder.dim * 5 + 4
        self.retrieval_cache = {}

    def allowed(self, item, online=False):
        k = self.config.get('top_k', 0)
        if not k:
            return None
        if self.retriever is None:
            raise RuntimeError('Retrieval arm requires an actual retriever')
        if not online and item['id'] in self.retrieval_cache:
            return self.retrieval_cache[item['id']]
        opts = item['catalog']['options']
        corpus = self.retriever.encode([option_text(o, True) for o in opts])
        query = self.retriever.query(item['request'], cached=not online)
        score = corpus @ query
        indices = torch.argsort(score, descending=True, stable=True)[:min(k, len(opts))].tolist()
        ranked = [opts[i]['id'] for i in indices]
        self.retrieval_cache[item['id']] = ranked
        return ranked

    def get(self, item, online=False):
        units = candidate_units(item, self.config['decoder'] == 'tree', self.config.get('typed', True))
        allowed = self.allowed(item, online)
        if allowed is not None:
            units = [u for u in units if u['id'] in SPECIAL or set(u['members']) & set(allowed)]
        q = self.encoder.query(item['request'], cached=not online)
        c = self.encoder.encode([u['text'] for u in units])
        q = q.expand_as(c)
        difference = c - c.mean(0, keepdim=True) if self.config.get('pairwise') else torch.zeros_like(c)
        state = torch.tensor([[len(u['path']) / 5, float(u['id'] in SPECIAL),
                               float(item['task'] == 'schema'), len(u['members']) / 8] for u in units])
        x = torch.cat([q, c, q * c, (q - c).abs(), difference, state], dim=-1)
        return x, units, allowed


class Head(nn.Module):
    def __init__(self, dim, adapter=False):
        super().__init__()
        self.adapter = nn.Sequential(nn.Linear(dim, 32), nn.GELU(), nn.Linear(32, dim)) if adapter else None
        self.score = nn.Sequential(nn.Linear(dim, 96), nn.GELU(), nn.Linear(96, 1))

    def forward(self, x):
        if self.adapter is not None:
            x = x + self.adapter(x)
        return self.score(x).squeeze(-1)


class Policy(nn.Module):
    def __init__(self, dim, mode='single', adapter=False):
        super().__init__()
        self.mode = mode
        count = 1 if mode in ['single', 'shared-policy'] else 2
        use_adapter = adapter or mode in ['separate-adapters', 'separate-checkpoints']
        self.heads = nn.ModuleList([Head(dim, use_adapter) for _ in range(count)])

    def forward(self, x, task):
        index = 0 if len(self.heads) == 1 or task == 'operation' else 1
        return self.heads[index](x)


def labels(reference, units):
    positive = set(reference['paths']) if reference['status'] == 'accepted' else {reference['status']}
    return torch.tensor([float(bool(positive & set(u['members']))) for u in units])


def decode(item, units, probabilities, threshold, tree=False):
    scores = {u['id']: float(v) for u, v in zip(units, probabilities)}
    sentinels = [key for key in SPECIAL if scores.get(key, 0) >= threshold]
    if sentinels:
        status = max(sentinels, key=lambda k: scores[k])
        return {'status': status, 'selected': [], 'scores': scores}
    selected = []
    for o in item['catalog']['options']:
        prefixes = ['.'.join(o['path'][:i]) for i in range(1, len(o['path']) + 1)] if tree else [o['id']]
        if all(scores.get(p, 0) >= threshold for p in prefixes):
            selected.append(o['id'])
    return {'status': 'accepted' if selected else 'NO_MATCH', 'selected': sorted(selected), 'scores': scores}


def predict(policy, feature_source, item, threshold, online=False):
    # Deliberately accepts PUBLIC item only. Reference answers are not arguments.
    x, units, allowed = feature_source.get(item, online=online)
    policy.eval()
    with torch.no_grad():
        probs = torch.sigmoid(policy(x, item['task']))
    result = decode(item, units, probs, threshold, feature_source.config['decoder'] == 'tree')
    result['retrieved'] = allowed
    return result


def state_hash(state):
    h = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        h.update(key.encode()); h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
