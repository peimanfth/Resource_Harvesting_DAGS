#! /bin/bash
set -e 
echo " starting redis installation and api gateway"
ansible-playbook -i environments/local apigateway.yml
echo " completed ansible playbook"