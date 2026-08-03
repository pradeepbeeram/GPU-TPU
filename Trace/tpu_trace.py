import urllib.request
import json
import time
import statistics
import argparse
import os
from datetime import datetime


def detect_infrastructure():
   """Automatically detects if running in TPU or GPU context, defaulting to TPU if ambiguous."""
   if os.path.exists("/dev/accel0") or "TPU_NAME" in os.environ or "NEXT_PLUGGABLE_DEVICE_DISABLE_CENTRAL_NODE" in os.environ:
       return "tpu"
   if os.path.exists("/dev/nvidia0") or "CUDA_VISIBLE_DEVICES" in os.environ:
       return "gpu"
   return "tpu"


# Hardware Peak FP16/BF16 TFLOPS defaults
PEAK_TFLOPS_DEFAULTS = {
   "tpu": 197.0,  # GCP TPU v5e (ct5lp-hightpu-1t)
   "gpu": 70.0    # GCP G2/N1 (NVIDIA T4 Tensor Core)
}


# --- Command Line Arguments ---
parser = argparse.ArgumentParser(description="Universal LLM benchmark script for TPU & GPU.")
parser.add_argument("--infra", type=str, choices=["tpu", "gpu", "auto"], default="auto", help="Infra override (default: auto-detect)")
parser.add_argument("--url", type=str, default="http://localhost:8000/v1/chat/completions", help="Endpoint URL")
parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Model name")
parser.add_argument("--params-b", type=float, default=0.49, help="Active parameters in billions")
parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling cutoff (0.0 to 1.0)")
parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling cutoff (-1 to disable)")
args = parser.parse_args()


infra_type = detect_infrastructure() if args.infra == "auto" else args.infra.lower()
url = args.url
model_name = args.model
params_in_billions = args.params_b
temperature = args.temperature
top_p = args.top_p
top_k = args.top_k
peak_hardware_tflops = PEAK_TFLOPS_DEFAULTS.get(infra_type, 70.0)


# Build parameter string for filenames
param_suffix = f"temp{temperature}_p{top_p}_k{top_k}"


categories = {
   'Coding': ['Write a binary search in Python.', 'Explain recursion with code.', 'Fix a memory leak in C++.'],
   'Science': ['How does photosynthesis work?', 'What is quantum entanglement?', 'How do solar panels work?'],
   'DevOps': ['Docker vs VMs summary.', 'Explain Kubernetes ingress rules.', 'What is GitOps?'],
   'Math': ['Derivative of x^2 * sin(x)?', 'Explain the Fibonacci sequence.', 'How do prime numbers work?'],
   'SQL': ['Query for 2nd highest salary.', 'INNER vs LEFT JOIN.', 'Explain indexing in Postgres.'],
   'Summarization': ['Summarize Hamlet in 3 sentences.', 'Summarize how the internet works.', 'Summarize machine learning.'],
   'Translation': ['Translate "Hello, how are you?" to French, Spanish, German.', 'Translate greeting to Japanese.'],
   'Creative': ['Write a poem about Cloud TPUs.', '2-sentence story about space.', 'Write a haiku about rain.'],
   'Business': ['3 bullet summary on customer retention.', 'Top metrics for SaaS startups.'],
   'Trivia': ['What are the 7 continents?', 'Element with atomic number 1?', 'Who painted the Mona Lisa?']
}


raw_prompts = []
for i in range(100):
   cat_keys = list(categories.keys())
   cat_name = cat_keys[i % len(cat_keys)]
   prompt_template = categories[cat_name][i % len(categories[cat_name])]
   raw_prompts.append(f'[{cat_name}] {prompt_template} (Trace #{i+1})')


ttfts = []
e2e_latencies = []
tokens_generated = []
successes = 0
failures = 0


print(f'=== STARTING 100-TRACE WORKLOAD ON [{infra_type.upper()}] ===')
print(f'Target Model : {model_name} ({params_in_billions}B params)')
print(f'Sampling Config: temp={temperature} | top_p={top_p} | top_k={top_k}')
print(f'Target URL   : {url}\n')


start_time = time.time()


for i, prompt in enumerate(raw_prompts):
   payload_dict = {
       'model': model_name,
       'messages': [{'role': 'user', 'content': prompt}],
       'max_tokens': 64,
       'temperature': temperature,
       'top_p': top_p,
       'top_k': top_k,
       'stream': True
   }
   data = json.dumps(payload_dict).encode('utf-8')
  
   req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
  
   t0 = time.time()
   first_token_time = None
   gen_tokens = 0
  
   try:
       with urllib.request.urlopen(req) as resp:
           for line in resp:
               line_str = line.decode('utf-8').strip()
               if line_str.startswith('data: ') and line_str != 'data: [DONE]':
                   if first_token_time is None:
                       first_token_time = time.time() - t0
                   gen_tokens += 1
          
           req_e2e_latency = time.time() - t0
          
           if first_token_time is not None:
               ttfts.append(first_token_time)
               e2e_latencies.append(req_e2e_latency)
               tokens_generated.append(gen_tokens)
               successes += 1
              
               if (i + 1) % 20 == 0 or i == 0:
                   print(f'[{infra_type.upper()} | Progress {i+1}/100] TTFT: {round(first_token_time*1000, 1)}ms | E2E Latency: {round(req_e2e_latency, 3)}s')
   except Exception as e:
       failures += 1
       print(f'[{infra_type.upper()} | Trace {i+1}] ERROR: {e}')


total_wall_time = time.time() - start_time
total_tokens = sum(tokens_generated)


# --- Throughput & RPS ---
avg_tps = total_tokens / total_wall_time if total_wall_time > 0 else 0
avg_rps = successes / total_wall_time if total_wall_time > 0 else 0


# --- TFLOPS & MFU Computation ---
total_params = params_in_billions * 1e9
flops_per_token = 2 * total_params
achieved_tflops = (avg_tps * flops_per_token) / 1e12
mfu_percent = (achieved_tflops / peak_hardware_tflops) * 100 if peak_hardware_tflops > 0 else 0.0


def calc_percentiles(data_list):
   if not data_list:
       return 0, 0
   data_list.sort()
   p50 = statistics.median(data_list)
   p95_idx = int(len(data_list) * 0.95) - 1
   p95 = data_list[max(0, p95_idx)]
   return p50, p95


ttft_p50, ttft_p95 = calc_percentiles(ttfts)
e2e_p50, e2e_p95 = calc_percentiles(e2e_latencies)


results_data = {
   "timestamp": datetime.utcnow().isoformat(),
   "infrastructure": infra_type.upper(),
   "model": model_name,
   "model_params_billions": params_in_billions,
   "sampling_config": {
       "temperature": temperature,
       "top_p": top_p,
       "top_k": top_k
   },
   "peak_hardware_tflops": peak_hardware_tflops,
   "total_requests": len(raw_prompts),
   "successes": successes,
   "failures": failures,
   "total_wall_time_sec": round(total_wall_time, 2),
   "total_tokens": total_tokens,
   "requests_per_sec": round(avg_rps, 2),
   "tokens_per_sec": round(avg_tps, 2),
   "achieved_tflops": round(achieved_tflops, 4),
   "mfu_percent": round(mfu_percent, 2),
   "ttft_p50_ms": round(ttft_p50 * 1000, 1),
   "ttft_p95_ms": round(ttft_p95 * 1000, 1),
   "e2e_latency_p50_sec": round(e2e_p50, 3),
   "e2e_latency_p95_sec": round(e2e_p95, 3)
}


# Dynamic filenames including infra and parameters
json_filename = f"results_{infra_type}_{param_suffix}.json"
txt_filename = f"results_{infra_type}_{param_suffix}.txt"


# Save structured JSON file
with open(json_filename, "w") as f:
   json.dump(results_data, f, indent=2)


summary_text = f"""
=============================================
     WORKLOAD TRACE SUMMARY [{infra_type.upper()}]
=============================================
Timestamp               : {results_data['timestamp']}
Model                   : {model_name} ({params_in_billions}B)
Sampling Config         : temp={temperature}, top_p={top_p}, top_k={top_k}
Total Requests Executed : {len(raw_prompts)}
Successful / Failed     : {successes} ok / {failures} err
Total Wall Time          : {round(total_wall_time, 2)} seconds
Total Tokens Generated  : {total_tokens} tokens
---------------------------------------------
Requests Per Second     : {round(avg_rps, 2)} req/s
Tokens Per Second       : {round(avg_tps, 2)} tok/s
---------------------------------------------
Achieved TFLOPS         : {round(achieved_tflops, 4)} TFLOPS
Hardware Peak TFLOPS    : {peak_hardware_tflops} TFLOPS
Model FLOPs Util (MFU)  : {round(mfu_percent, 2)} %
---------------------------------------------
TTFT P50 (Median)       : {round(ttft_p50*1000, 1)} ms ({round(ttft_p50, 3)}s)
TTFT P95                : {round(ttft_p95*1000, 1)} ms ({round(ttft_p95, 3)}s)
---------------------------------------------
E2E Latency P50 (Median): {round(e2e_p50, 3)}s ({round(e2e_p50*1000, 1)} ms)
E2E Latency P95         : {round(e2e_p95, 3)}s ({round(e2e_p95*1000, 1)} ms)
=============================================
"""


print(summary_text)


# Save text summary report
with open(txt_filename, "w") as f:
   f.write(summary_text)


print(f"\nSaved results to '{json_filename}' and '{txt_filename}'.")
