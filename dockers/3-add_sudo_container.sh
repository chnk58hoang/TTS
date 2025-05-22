if [ -z $1 ]; then
    echo "==========================================================================================="
    echo "container id/name must not empty! run: bash add_sudo_for_user_asr.sh [container_id_or_name]"
    echo "=====================================list container========================================"
    docker ps
    echo "==========================================================================================="
else
    echo "seting up sudo for user asr in docker container: $1 ..."
    docker exec -it --user root $1 /bin/bash -c "echo '%asr ALL=(ALL:ALL) ALL'>>/etc/sudoers && apt-get update && apt install sudo"
    echo "done. now you can use sudo in container: $1"
    echo "attach into container: $1"
    docker exec -it $1 /bin/bash
fi