"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py

MATHEMATICAL UPGRADES:
- Hebbian Learning: Plastic synaptic connections that strengthen with use
- Hopfield Memory: Associative memory patterns with circular buffer storage
- Pi-Delta Oscillation: Rhythmic modulation of neural activity
- Infinity Gating: Dynamic thresholding mechanism
- Experience Buffer: Episodic memory for recent interactions
"""

import math
import inspect
from dataclasses import dataclass
from collections import deque

import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken

# Placeholder for a special tool token
TOOL_TOKEN_ID = 50257

class ExperienceBuffer:
    """Ring buffer to store recent model interactions for episodic memory"""
    
    def __init__(self, maxlen=512):
        self.buffer = deque(maxlen=maxlen)
    
    def push(self, x, y):
        """Store input-target pairs"""
        self.buffer.append((x.detach().cpu(), y.detach().cpu()))
    
    def sample(self, n=5):
        """Sample recent experiences"""
        if len(self.buffer) < n:
            return list(self.buffer)
        return list(self.buffer)[-n:]
    
    def __len__(self):
        return len(self.buffer)


class HebbianLayer(nn.Module):
    """Plastic neural layer implementing Hebbian learning: 'neurons that fire together, wire together'"""
    
    def __init__(self, size, decay=0.99, lr=0.001):
        super().__init__()
        self.size = size
        self.decay = decay
        self.lr = lr
        # plastic weight as buffer
        self.register_buffer('plastic_weights', torch.zeros(size, size))
        self.fixed_weights = nn.Parameter(torch.randn(size, size) * 0.02)
        
    def forward(self, x):
        # x: (batch_size, seq_len, features)
        batch_size, seq_len, features = x.shape

        # Hebbian update: strengthen connections based on co-activation
        if self.training:
            with torch.no_grad():
                x_flat = x.view(-1, features)
                correlation = torch.mm(x_flat.t(), x_flat) / x_flat.shape[0]
                self.plastic_weights.mul_(self.decay).add_(self.lr * correlation)

        # Apply both fixed and plastic transformations
        fixed_out = torch.matmul(x, self.fixed_weights)
        plastic_out = torch.matmul(x, self.plastic_weights)

        return fixed_out + plastic_out


class HopfieldLayer(nn.Module):
    """Hopfield associative memory with circular buffer for pattern storage"""
    
    def __init__(self, embed_dim, num_patterns=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patterns = num_patterns
        self.pattern_idx = 0
        
        # Memory matrix to store patterns
        self.register_buffer('memory', torch.randn(num_patterns, embed_dim) * 0.02)
        # Learnable query and key projections
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        
    def store(self, x):
        """Store new patterns in circular buffer"""
        if self.training:
            batch_size, seq_len, _ = x.shape
            # Use mean pooling to get representative pattern
            pattern = x.mean(dim=(0, 1))  # (embed_dim,)
            
            with torch.no_grad():
                self.memory.data[self.pattern_idx] = pattern
                self.pattern_idx = (self.pattern_idx + 1) % self.num_patterns
                
        return x
    
    def recall(self, x):
        """Recall similar patterns from memory"""
        batch_size, seq_len, embed_dim = x.shape
        
        # Project inputs
        queries = self.query_proj(x)  # (batch, seq, embed)
        keys = self.key_proj(self.memory.unsqueeze(0).expand(batch_size, -1, -1))  # (batch, patterns, embed)
        
        # Compute attention scores
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(embed_dim)
        attention = F.softmax(scores, dim=-1)
        
        # Retrieve patterns
        retrieved = torch.matmul(attention, keys)
        
        return retrieved

class LayerNorm(nn.Module):
    """ Layer Norm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        
        # Mathematical enhancement components
        self.hebbian = HebbianLayer(config.n_embd, decay=0.99, lr=0.001)
        self.hopfield = HopfieldLayer(config.n_embd, num_patterns=128)

    def forward(self, x):
        # Standard MLP pathway
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        
        # Mathematical enhancement pathway: Hebbian -> Store -> Recall
        x_hebbian = self.hebbian(x)
        x_stored = self.hopfield.store(x_hebbian)  # Store patterns (returns input unchanged)
        x_recalled = self.hopfield.recall(x_stored)  # Recall similar patterns
        
        # Combine original and enhanced pathways
        return x + x_recalled

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    
    # Mathematical enhancement parameters
    pi_freq: float = 0.1  # Pi-Delta oscillation frequency
    pi_amp: float = 0.05  # Pi-Delta oscillation amplitude
    infinity_threshold: float = 0.8  # Gating threshold for infinity mechanism
    emotion_vector_size: int = 16 # Size of the emotion vector

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Mathematical enhancement components
        self.mem_token = nn.Parameter(torch.zeros(1, 1, config.n_embd))
        self.experience_buffer = ExperienceBuffer(maxlen=512)
        
        # Emotional State Vector
        self.emotion_vector = nn.Parameter(torch.zeros(1, 1, config.vocab_size))
        
        # Simple sentiment dictionary
        enc = tiktoken.get_encoding("gpt2")
        positive_words = ["happy", "joy", "good", "love", "great", "wonderful", "beautiful", "nice"]
        negative_words = ["sad", "bad", "hate", "cry", "terrible", "awful", "horrible", "ugly"]
        self.sentiment_map = {
            "positive": [enc.encode(w)[0] for w in positive_words],
            "negative": [enc.encode(w)[0] for w in negative_words],
        }

        # init all weights
        self.apply(self._init_weights)

        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)

        if t < self.config.block_size:
            # pos for real tokens should start at 1 when mem token is at 0
            pos = torch.arange(1, t+1, dtype=torch.long, device=device)
            pos_emb = self.transformer.wpe(pos)  # (t, n_embd)
            embeddings = tok_emb + pos_emb
            mem_token_expanded = self.mem_token.expand(b, 1, -1)
            x = torch.cat([mem_token_expanded, embeddings], dim=1)
        else:
            pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
            pos_emb = self.transformer.wpe(pos)
            embeddings = tok_emb + pos_emb
            x = embeddings

        x = self.transformer.drop(x)
        
        # Forward through transformer blocks (enhanced with Hebbian/Hopfield in MLP)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        
        # Remove memory token from output if it was added
        if t < self.config.block_size and x.size(1) == t + 1:
            x = x[:, 1:, :]  # Remove first token (memory token)

        if targets is not None:
            # Training mode: calculate logits and loss
            logits = self.lm_head(x)
            
            # Apply Pi-Delta Oscillation and Infinity Gating
            logits = self._apply_mathematical_enhancements(logits)
            
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference mode: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            
            # Apply Pi-Delta Oscillation and Infinity Gating
            logits = self._apply_mathematical_enhancements(logits)
            
            loss = None

        return logits, loss
    
    def _update_emotion_vector(self, token_id):
        if token_id in self.sentiment_map["positive"]:
            self.emotion_vector[:, :, token_id] += 0.1
        elif token_id in self.sentiment_map["negative"]:
            self.emotion_vector[:, :, token_id] -= 0.1

    def _apply_mathematical_enhancements(self, logits):
        """Apply Pi-Delta Oscillation and Infinity Gating to logits"""
        # Pi-Delta Oscillation: rhythmic modulation of neural activity
        time_step = getattr(self, '_time_step', 0)
        self._time_step = time_step + 1
        
        pi_modulation = self.config.pi_amp * torch.sin(
            self.config.pi_freq * time_step * torch.ones_like(logits)
        )
        logits = logits + pi_modulation
        
        # Infinity Gating: dynamic thresholding mechanism
        # Amplify logits that exceed the threshold, suppress others
        max_logits = torch.max(logits, dim=-1, keepdim=True)[0]
        normalized_logits = logits / (max_logits + 1e-8)  # Normalize to prevent overflow
        
        gate = torch.sigmoid(10 * (normalized_logits - self.config.infinity_threshold))
        logits = logits * gate + normalized_logits * (1 - gate)
        
        # Add emotional state
        logits = logits + self.emotion_vector

        # [LATENT HOOK FOR TOOL CALLING]
        # In a future update, we can check if the top logit corresponds to a special [TOOL] token.
        # If so, we can parse the generated text for a tool request and call an external API.
        # This hook provides the integration point without adding complexity now.
        if torch.argmax(logits[:, -1, :]) == TOOL_TOKEN_ID:
            print("TOOL CALL DETECTED!")
            # Logic to handle tool call would go here
            pass
        
        return logits

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer with fused option if available and on CUDA
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # Update emotional state
            self._update_emotion_vector(idx_next.item())
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx