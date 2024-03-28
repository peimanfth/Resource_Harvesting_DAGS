#! /bin/bash
set -e

ansible-playbook -i environments/local couchdb.yml -e mode=clean


ansible-playbook -i environments/local couchdb.yml
ansible-playbook -i environments/local initdb.yml
ansible-playbook -i environments/local wipe.yml


