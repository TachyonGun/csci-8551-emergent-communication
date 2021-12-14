max_agents=$1
max_landmarks=$2
prefix=$3

for num_agents in $(seq 2 $max_agents); do
  for num_landmarks in $(seq 2 $max_landmarks); do
    echo "python train.py --max_agents $num_agents --min_agents $num_agents --min_landmarks $num_landmarks --max_landmarks $num_landmarks --prefix $prefix/agents_$num_agents/landmarks_$num_landmarks/"
    mkdir -p $prefix/agents_$num_agents/landmarks_$num_landmarks/
    python train.py --max-agents $num_agents --min-agents $num_agents --min-landmarks $num_landmarks --max-landmarks $num_landmarks --prefix $prefix/agents_$num_agents/landmarks_$num_landmarks/ --n-epochs 50
  done
done