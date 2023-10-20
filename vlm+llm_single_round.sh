# caption
python scripts/caption.py --vlm blip2
python scripts/caption.py --vlm instructBLIP

# BLIP-2 + ChatGPT
python scripts/vlm_llm_single.py --vlm blip2 --llm chatgpt
python scripts/eval_vlm_llm.py --vlm blip2 --llm chatgpt

# instructBLIP + ChatGPT
python scripts/vlm_llm_single.py --vlm instructBLIP --llm chatgpt
python scripts/eval_vlm_llm.py --vlm instructBLIP --llm chatgpt

# BLIP-2 + GPT-4
python scripts/vlm_llm_single.py --vlm blip2 --llm gpt4
python scripts/eval_vlm_llm.py --vlm blip2 --llm gpt4

# instructBLIP + GPT-4
python scripts/vlm_llm_single.py --vlm instructBLIP --llm gpt4
python scripts/eval_vlm_llm.py --vlm instructBLIP --llm gpt4