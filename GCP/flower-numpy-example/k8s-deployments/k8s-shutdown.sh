#! /bin/bash -l

# Change directory to the yaml files directory
cd "$(dirname "${BASH_SOURCE[0]}")"

kubectl delete -f superlink-deployment.yaml
sleep 0.1

kubectl delete -f supernode-1-deployment.yaml
sleep 0.1

kubectl delete -f supernode-2-deployment.yaml
sleep 0.1

kubectl delete -f superexec-serverapp-deployment.yaml
sleep 0.1

kubectl delete -f superexec-clientapp-1-deployment.yaml
sleep 0.1

kubectl delete -f superexec-clientapp-2-deployment.yaml
sleep 0.1