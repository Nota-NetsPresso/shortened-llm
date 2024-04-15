import argparse
import os
from transformers import AutoModelForCausalLM
from utils import set_seed, get_model, get_block_pruned_network
from eval_ppl import eval_ppl_wikitext2_ptb, generate_txt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, default="baffo32/decapoda-research-llama-7B-hf", help='base model name')
    parser.add_argument('--tokenizer',type=str, default=None, help='if None, base model name is used')
    parser.add_argument('--model_type', type=str, default="pretrain", choices=['pretrain', 'pruneLLM', 'tune_pruneLLM'])
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--lora_ckpt', type=str, default=None)
    parser.add_argument('--device', type=str, default="cuda", help='device')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--num_pruned_blocks', type=int, default=6)     
    parser.add_argument('--block_order_csv', type=str, default="results/llama-7b-hf/block_sensitivity_n10/block_order_ppl.csv")
    parser.add_argument('--output_dir', type=str, default="output_prune/llama-7b-hf/rm_6_blocks")
    parser.add_argument('--skip_validation', default=False, action="store_true")
    parser.add_argument('--no_plus_heuristic', default=False, action="store_true")
    parser.add_argument('--fix_decapoda_config', default=False, action="store_true", help="fix tokenizer config of baffo32/decapoda-research-llama-7B-hf")
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    model_orig, tokenizer, description = get_model(base_model=args.base_model, ckpt=args.ckpt, lora_ckpt=args.lora_ckpt,
                                            tokenizer=args.tokenizer, model_type=args.model_type, device='cpu',
                                            fix_decapoda_config=args.fix_decapoda_config)    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the precomputed block unimportance order    
    unimportance_order = []
    with open(args.block_order_csv, 'r') as file:
        unimportance_order = [int(i) for i in str(next(file).strip()).split(',')]
    if not args.no_plus_heuristic:
        last_block_index=model_orig.config.num_hidden_layers-1
        keep_block_info = [0,1,2,3,last_block_index-1,last_block_index] # to keep first and last few blocks unpruned
        unimportance_order = [idx for idx in unimportance_order if idx not in keep_block_info]     
    
    # Block-level pruning 
    model = get_block_pruned_network(model_orig,
                                     unimportance_order=unimportance_order,
                                     num_pruned_blocks=args.num_pruned_blocks,
                                     device=args.device)    
    
    # Save     
    model.save_pretrained(args.output_dir, max_shard_size="10GB") 
    tokenizer.save_pretrained(args.output_dir)  
     
    # Measure PPL     
    if not args.skip_validation:
        score_dir = os.path.join(args.output_dir+'_score')
        os.makedirs(score_dir, exist_ok=True)
        eval_ppl_wikitext2_ptb(output_dir=score_dir, model=model, tokenizer=tokenizer, device=args.device)
        generate_txt(output_dir=score_dir, model=model, tokenizer=tokenizer, device=args.device)