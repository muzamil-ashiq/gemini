"""
Enhanced Sample from a trained model with Experience Buffer
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file

# Enhanced Experience Buffer for Memory-Aware Generation
class EnhancedExperienceBuffer:
    def __init__(self, capacity=20, embedding_dim=256):
        self.capacity = capacity
        self.experiences = []
        self.embedding_dim = embedding_dim
        self.generation_count = 0
        
    def add_experience(self, text, hidden_state=None, quality_score=1.0):
        """Add a new experience with text and optional hidden state"""
        experience = {
            'text': text,
            'hidden_state': hidden_state.detach().clone() if torch.is_tensor(hidden_state) else None,
            'importance': quality_score,
            'generation_id': self.generation_count,
            'coherence_score': self._calculate_coherence(text)
        }
        
        if len(self.experiences) >= self.capacity:
            # Remove least important experience
            self.experiences.sort(key=lambda x: x['importance'] * x['coherence_score'])
            self.experiences.pop(0)
            
        self.experiences.append(experience)
        
        # Update importance scores
        for exp in self.experiences:
            # Decay older experiences
            age_factor = 1.0 - (self.generation_count - exp['generation_id']) * 0.05
            exp['importance'] *= max(0.1, age_factor)
    
    def _calculate_coherence(self, text):
        """Simple coherence scoring based on text length and structure"""
        if not text or len(text) < 10:
            return 0.1
        
        # Basic coherence indicators
        has_proper_ending = any(text.endswith(punct) for punct in ['.', '!', '?'])
        word_count = len(text.split())
        
        coherence = 0.5
        if has_proper_ending:
            coherence += 0.3
        if word_count > 20:
            coherence += 0.2
        
        return min(1.0, coherence)
    
    def get_context_prompt(self, max_context_length=50):
        """Get context from high-quality experiences"""
        if not self.experiences:
            return ""
            
        # Sort by quality (importance * coherence)
        quality_experiences = sorted(
            self.experiences, 
            key=lambda x: x['importance'] * x['coherence_score'], 
            reverse=True
        )
        
        context_parts = []
        total_length = 0
        
        for exp in quality_experiences[:3]:  # Top 3 experiences
            text = exp['text'][:100]  # Limit length
            if total_length + len(text) < max_context_length:
                context_parts.append(text)
                total_length += len(text)
            else:
                break
                
        return " ".join(context_parts)
    
    def get_memory_influence(self):
        """Get memory influence factor for generation"""
        if not self.experiences:
            return 0.0
        
        avg_coherence = sum(exp['coherence_score'] for exp in self.experiences) / len(self.experiences)
        return min(0.3, avg_coherence * 0.5)  # Up to 30% influence
    
    def __len__(self):
        return len(self.experiences)

# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# Initialize enhanced experience buffer
experience_buffer = EnhancedExperienceBuffer(capacity=25, embedding_dim=model.config.n_embd)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()

# Enhanced generation with memory
print(f"🧠 Enhanced Memory-Aware Generation")
print(f"Mathematical model: {sum(p.numel() for p in model.parameters()):,} parameters")
print(f"Experience buffer capacity: {experience_buffer.capacity}")
print("=" * 60)

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            # Incorporate memory context if available
            context_prompt = experience_buffer.get_context_prompt(max_context_length=30)
            memory_influence = experience_buffer.get_memory_influence()
            
            # Modify prompt with context if we have good memories
            effective_start = start
            if context_prompt and memory_influence > 0.1:
                effective_start = f"{context_prompt} {start}"
                print(f"🔗 Using context from {len(experience_buffer)} memories (influence: {memory_influence:.1%})")
            
            start_ids = encode(effective_start)
            x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
            
            # Adjust temperature based on memory confidence
            adjusted_temp = temperature * (1.0 - memory_influence * 0.3)
            
            print(f"=== Enhanced Sample {k+1}/{num_samples} ===")
            print(f"Experience buffer contains {len(experience_buffer)} memories")
            print(f"Adjusted temperature: {adjusted_temp:.2f}")
            
            y = model.generate(x, max_new_tokens, temperature=adjusted_temp, top_k=top_k)
            generated_text = decode(y[0].tolist())
            
            # Extract only the new generated part
            original_length = len(decode(start_ids))
            if context_prompt:
                # Remove context from display
                context_length = len(context_prompt) + 1
                display_text = generated_text[context_length + original_length:]
                new_generated = generated_text[context_length:]
            else:
                display_text = generated_text[original_length:]
                new_generated = generated_text
            
            print(f"{start}{display_text}")
            
            # Add this generation to experience buffer with quality scoring
            quality_score = 1.0
            if len(display_text) > 50:
                quality_score += 0.2
            if any(punct in display_text for punct in ['.', '!', '?']):
                quality_score += 0.3
            
            experience_buffer.add_experience(new_generated, quality_score=quality_score)
            experience_buffer.generation_count += 1
            
            print('---------------')

print(f"\n🎯 FINAL EXPERIENCE BUFFER STATE:")
print(f"Total experiences accumulated: {len(experience_buffer)}")
print(f"Average coherence score: {sum(exp['coherence_score'] for exp in experience_buffer.experiences) / len(experience_buffer.experiences) if experience_buffer.experiences else 0:.2f}")
print("The enhanced experience buffer now contains contextual memories that influence")
print("future generations, enabling more coherent and contextually aware responses!")
