#!/usr/bin/env python3
"""
Process conversational data and merge with TinyStories for enhanced dialogue training
"""
import tiktoken
import numpy as np
import pickle
import os

def process_conversation_data():
    print("Processing conversational data for enhanced dialogue training...")
    
    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    # Read conversation data
    with open('conversation_data.txt', 'r', encoding='utf-8') as f:
        conversations = f.readlines()
    
    # Process conversations
    conversation_tokens = []
    for conv in conversations:
        conv = conv.strip()
        if conv:
            # Add special tokens to mark dialogue
            conv = "<|dialogue|>" + conv + "<|enddialogue|>"
            tokens = enc.encode(conv)
            conversation_tokens.extend(tokens)
    
    print(f"Processed {len(conversations)} conversations")
    print(f"Generated {len(conversation_tokens):,} conversation tokens")
    
    # Load existing TinyStories data
    print("Loading existing TinyStories data...")
    existing_train = np.memmap('train.bin', dtype=np.uint16, mode='r')
    existing_val = np.memmap('val.bin', dtype=np.uint16, mode='r')
    
    print(f"Existing training tokens: {len(existing_train):,}")
    print(f"Existing validation tokens: {len(existing_val):,}")
    
    # Create enhanced dataset
    # Add conversations multiple times to increase their weight in training
    conversation_multiplier = 5  # Repeat conversations 5x for emphasis
    enhanced_conversations = conversation_tokens * conversation_multiplier
    
    print(f"Enhanced conversation tokens (5x): {len(enhanced_conversations):,}")
    
    # Combine datasets - conversations first for priority learning
    enhanced_train = np.array(enhanced_conversations + existing_train.tolist(), dtype=np.uint16)
    enhanced_val = np.array(enhanced_conversations[:len(enhanced_conversations)//10] + existing_val.tolist(), dtype=np.uint16)
    
    # Save enhanced datasets
    print("Saving enhanced datasets...")
    
    # Backup original data
    os.rename('train.bin', 'train_original.bin')
    os.rename('val.bin', 'val_original.bin')
    
    # Save enhanced data
    with open('train.bin', 'wb') as f:
        enhanced_train.tofile(f)
    
    with open('val.bin', 'wb') as f:
        enhanced_val.tofile(f)
    
    # Update meta.pkl if needed
    meta_path = 'meta.pkl'
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        
        # Add special dialogue tokens to vocabulary if needed
        if 'stoi' in meta and 'itos' in meta:
            print("Meta file contains character-level encoding - keeping as is")
        else:
            print("Meta file is token-level - updating with dialogue tokens")
    
    print(f"Enhanced dataset created!")
    print(f"New training tokens: {len(enhanced_train):,} (+{len(enhanced_conversations):,} conversation)")
    print(f"New validation tokens: {len(enhanced_val):,}")
    print(f"Conversation data weight: {conversation_multiplier}x multiplier")
    
    return len(enhanced_train), len(enhanced_val)

if __name__ == "__main__":
    train_size, val_size = process_conversation_data()
    
    # Calculate new epoch information
    tokens_per_iter = 40 * 8 * 256  # gradient_accumulation_steps * batch_size * block_size
    iters_per_epoch = int(train_size / tokens_per_iter)
    
    # For 2-hour training
    seconds_per_iter = 11  # From previous measurements
    hours = 2
    max_iterations = (hours * 3600) // seconds_per_iter
    max_epochs = max_iterations // iters_per_epoch
    
    print(f"\n📊 2-Hour Training Plan:")
    print(f"Iterations per epoch: {iters_per_epoch}")
    print(f"Maximum iterations in 2 hours: {max_iterations}")
    print(f"Maximum epochs in 2 hours: ~{max_epochs}")
    print(f"Training will include enhanced dialogue capabilities!")
