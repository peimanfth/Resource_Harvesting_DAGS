##! /bin/bash
set -e
cd ../ansible

ansible-playbook -i environments/local wipe.yml

cd ../front-end
# ansible-playbook -i environments/local initdb.yml


