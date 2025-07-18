# Profiling Visualization

Following our [profile guidance](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/tool_examples/debug-tp-2.md), the log is stored in:

```bash
"logs/"+os.getenv("EXPERIMENT_NAME", "multiturn_log_dir"),
    f"step_{self.step}",
    f"worker_{self._rank}.jsonl"
```

in the format of

```
step_0/
    worker_0.jsonl
    worker_1.jsonl
    ...
step_1/
    worker_0.jsonl
    worker_1.jsonl
```

# Analysis

We've provided some scripts to analyze the logs. Our visualized results are in [script-examples](./script-examples):

1. `cdf_per_step.py`: Collects all requests from all workers within each step, and plots the CDF of the request duration.

2. `overview_duration.py`: Plots the rollout and training duration for each step, shows step-to-step timing trends and patterns.

3. `per_step_all_workers.py`: Analyzes time distribution for each step across all workers, shows load balancing and worker performance variations, identifies the slowest worker per step (critical path analysis), and visualizes worker synchronization patterns. 

4. `req_analysis_and_cdf.py`: workers on the log of a specific worker in a specific step, and plot the cdf of the request duration.
plot the top slowest requests, breaks down individual requests into processing phases, and shows detailed phase-by-phase timing breakdown

5. `turn_difference.py`: Plot the average engine call duration of each turn.

6. `turn_distribution.py`: plot the number of turns distribution of each step.

7. `event_persentage_step.py`: plot the event time taken in percentage of each step.


# A Quick Reproduction to profile

```bash
docker run \
    -it \
    --shm-size 32g \
    --gpus all \
    -v {YOUR_HUGGINGFACE_CACHE_DIR}:/root/.cache \
    --ipc=host \
    --network=host \
    --privileged \
    --name sglang-FSDP-TP-2 \
    lmsysorg/sglang:dev \
    /bin/zsh
```

```bash
apt update
apt install -y python3.10 python3.10-venv
python3 -m ensurepip --upgrade

python3 -m venv ~/.python/veRL-multiturn-rollout

source ~/.python/veRL-multiturn-rollout/bin/activate

python3 -m pip install uv
```

```bash
cd ~
git clone -b multiturn_profile_log https://github.com/PrinsYin/verl.git
cd verl

python3 -m uv pip install -e ".[sglang]"

python3 -m uv pip install wheel
python3 -m uv pip install packaging
# best to use for torch 2.6.0 and sglang 0.4.6.post1 flash-attn 2.7.4.post1
python3 -m uv pip install flash-attn==2.7.4.post1 --no-build-isolation --no-deps

python3 -m uv pip install .
python3 -m uv pip install -r ./requirements.txt
```

```bash
export WANDB_API_KEY={YOUR_WANDB_API_KEY}

function now() {
    date '+%Y-%m-%d-%H-%M'
}
```

```bash
python3 ./examples/data_preprocess/gsm8k_multiturn_w_tool.py

bash profile-sglang-multi-turn.sh 1 
# The final 1 is the number of SGLang TP size.
```
