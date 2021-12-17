min_agents=$1
max_agents=$2
min_landmarks=$3
max_landmarks=$4
prefix=$5
epochs=$6
vocab_size_min=$7
vocab_size_max=$8

for vocab_size in $(seq $vocab_size_min $vocab_size_max); do
  for num_agents in $(seq $min_agents $max_agents); do
    for num_landmarks in $(seq $min_landmarks $max_landmarks); do
      echo "python train.py --max_agents $num_agents --min_agents $num_agents --min_landmarks $num_landmarks --max_landmarks $num_landmarks --prefix $prefix/agents_$num_agents/landmarks_$num_landmarks/ --n-epochs $epochs"
      mkdir -p $prefix/vocab_$vocab_size/agents_$num_agents/landmarks_$num_landmarks/
      python train.py --max-agents $num_agents --min-agents $num_agents --min-landmarks $num_landmarks --max-landmarks $num_landmarks --prefix $prefix/agents_$num_agents/landmarks_$num_landmarks --n-epochs $epochs --show-utterances --vocab-size $vocab_size
    done
  done
done