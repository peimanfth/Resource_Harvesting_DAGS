#! /bin/bash
cd ..
set -e
./gradlew distDocker
cd ./ansible
echo " completed gradle build"
ansible-playbook -i environments/local openwhisk.yml
echo " completed ansible build"