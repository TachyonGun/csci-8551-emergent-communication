max_agents=$1
max_landmarks=$2


for num_agents in $(seq 2 $max_agents); do
  for num_landmarks in $(seq 2 $max_landmarks); do
    python train.py --max_agents $num_agents --min_agents $num_agents --min_landmarks $num_landmarks --max_landmarks $num_landmarks
  done
done