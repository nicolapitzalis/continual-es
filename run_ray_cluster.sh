#!/bin/bash
#SBATCH --job-name=ray_es_cluster
#SBATCH --nodes=4
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out

# Activate the virtual environment
eval "$(conda shell.bash hook)"
conda activate es
echo "Virtual environment activated."
echo "Starting SLURM job on $(hostname)"

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --block &

    # optional, though may be useful in certain versions of Ray < 1.0.
sleep 10

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --block &
    sleep 5
done

start=$(date +%s)
python -u train_es.py \
    --ray-address $ip_head \
    --envs CartPole-v1 \
    --to-train 0 \
    --sigma 0.1 \
    --alpha 0.03 \
    --hidden-dims 64 64 \
    --iterations 500 \
    --num-workers 12 \
    --batch-size 32 \
    --weight-decay 0.0 \
    --rank-function centered \
    --checkpoint-interval 0 \
    --shared-output False 
    # --checkpoint /home/n.pitzalis/es/chkpts/best_policy_log_Hopper-v5_s0.1_a0.05_i1000_b32_w0.0_centered_amsTrue_Walker2d-v5_replay0_shared_output_frozen_hidden.pth \
    # --frozen-hidden True 
    # --adaptive-max-steps True \
    # --fwt True \
    # --replay-batch-size 24 \
    # --replay-weight 1.0 
    
end=$(date +%s)
echo "Elapsed time: $((end - start)) seconds"

