#! /bin/bash -l

# Change directory to the yaml files directory
cd "$(dirname "${BASH_SOURCE[0]}")"

kubectl apply -f superlink-deployment.yaml
sleep 0.1

kubectl apply -f supernode-1-deployment.yaml
sleep 0.1

kubectl apply -f supernode-2-deployment.yaml
sleep 0.1

kubectl apply -f superexec-serverapp-deployment.yaml
sleep 0.1

kubectl apply -f superexec-clientapp-1-deployment.yaml
sleep 0.1

kubectl apply -f superexec-clientapp-2-deployment.yaml
sleep 0.1