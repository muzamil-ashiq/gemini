#!/usr/bin/env python3
"""
Debug script to diagnose the AdamW fused optimizer issue
"""
import torch
import inspect
from model import GPT, GPTConfig

def debug_optimizer_issue():
    """Debug the fused AdamW parameter group problem"""
    
    # Create a model like in training
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(1337)
    
    config = GPTConfig(
        block_size=64,
        vocab_size=50257, 
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.0,
        bias=False,
    )
    
    model = GPT(config)
    model.to(device)
    
    print(f"🔧 DEBUGGING FUSED ADAMW ISSUE")
    print(f"Device: {device}")
    print(f"Model parameters: {model.get_num_params():,}")
    print()
    
    # Configure optimizer like in training
    weight_decay = 1e-1
    learning_rate = 1e-4
    betas = (0.9, 0.95)
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
    
    optimizer = model.configure_optimizers(
        weight_decay=weight_decay,
        learning_rate=learning_rate, 
        betas=betas,
        device_type=device_type
    )
    
    print("📊 PARAMETER GROUP ANALYSIS:")
    for i, group in enumerate(optimizer.param_groups):
        params = group['params']
        if params:
            print(f"Group {i} ({group.get('group_name', 'unnamed')}):")
            print(f"  Parameters: {len(params)}")
            print(f"  Learning rate: {group['lr']}")
            print(f"  Weight decay: {group['weight_decay']}")
            print(f"  First param dtype: {params[0].dtype}")
            print(f"  First param device: {params[0].device}")
            print(f"  First param shape: {params[0].shape}")
            print()
    
    # Test if the issue is multiple parameter groups
    print("🧪 TESTING SOLUTIONS:")
    
    # Solution 1: Single parameter group (disable math enhancements optimization)
    try:
        all_params = []
        for group in optimizer.param_groups:
            all_params.extend(group['params'])
        
        single_group_optimizer = torch.optim.AdamW(
            all_params, 
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
            fused=True
        )
        print("✓ Single group fused AdamW: SUCCESS")
    except Exception as e:
        print(f"✗ Single group fused AdamW: {e}")
    
    # Solution 2: Use non-fused for multiple groups
    try:
        multi_group_optimizer = torch.optim.AdamW(
            optimizer.param_groups,
            betas=betas,
            fused=False
        )
        print("✓ Multi-group non-fused AdamW: SUCCESS") 
    except Exception as e:
        print(f"✗ Multi-group non-fused AdamW: {e}")
    
    # Solution 3: Check if all params have same dtype
    all_same_dtype = True
    all_same_device = True
    first_dtype = None
    first_device = None
    
    for group in optimizer.param_groups:
        for param in group['params']:
            if first_dtype is None:
                first_dtype = param.dtype
                first_device = param.device
            elif param.dtype != first_dtype or param.device != first_device:
                all_same_dtype = False
                all_same_device = False
                break
    
    print(f"All parameters same dtype: {all_same_dtype}")
    print(f"All parameters same device: {all_same_device}")
    
    return optimizer

if __name__ == "__main__":
    debug_optimizer_issue()
