# config for fine-tuning on tool_data
# this is a tiny model, so it should run on a 4GB GPU

out_dir = 'out-finetune-tools'
eval_interval = 200 # keep frequent so we can see progress
eval_iters = 100
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so smaller learning rate and fewer iterations
learning_rate = 1e-4 # reduced learning rate for fine-tuning
max_iters = 2000 # total number of training iterations
lr_decay_iters = 2000 # make learning rate decay throughout fine-tuning
min_lr = 1e-5 # learning rate will decay to this value

# data
dataset = 'tool_data'
batch_size = 8 # TINY MODEL: Small batch size for 4GB VRAM
block_size = 256 # TINY MODEL: Reduced block size for 4GB VRAM

# model
n_layer = 4 # TINY MODEL: Reduced layers
n_head = 4 # TINY MODEL: Reduced attention heads
n_embd = 256 # TINY MODEL: Reduced embedding dimension

# these make the total batch size be 8*8 = 64
gradient_accumulation_steps = 8

# always use cuda
device = 'cuda'
compile = False # don't compile for fine-tuning on small dataset