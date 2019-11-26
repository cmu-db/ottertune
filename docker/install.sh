#!/bin/bash

service="$1"


echo ""
if [ -z "$service" ] || ([ "$service" != "web" ] && [ "$service" != "driver" ]) 
then
    echo "Invalid value for service: '$service'"
    echo ""
    echo "Usage: $0 [web|driver]"
    exit 1
fi

echo ""
echo "-=------------------------------------------------------"
echo " Starting installation for service '$service'..."
echo "-=------------------------------------------------------"

if [ "$DEBUG" = true ]
then
    echo ""
    echo "Environment Variables:"
    echo " - DEBIAN_FRONTEND: $DEBIAN_FRONTEND"
    echo " - GRADLE_VERSION:  $GRADLE_VERSION"
    echo " - GRADLE_HOME:     $GRADLE_HOME"
    echo " - PATH:            $PATH"
    echo ""
fi

apt_pkgs="python3.6 python3-setuptools python3-pip libssl-dev git"
rm_pkgs=""
install_gradle=false
pip_reqs=/requirements.txt

if [ "$service" = "web" ]
then
    apt_pkgs="$apt_pkgs python3-dev gcc mysql-client libmysqlclient-dev python-mysqldb postgresql-client"

    rm_pkgs="$rm_pkgs gcc"

else
    apt_pkgs="$apt_pkgs openssh-server openjdk-11-jdk checkstyle unzip wget"

    # Hack: filter driver pip dependencies
    >tmp.txt
    for pip_pkg in autopep8 Fabric3 numpy requests pycodestyle pylint git-lint
    do
        grep "^$pip_pkg" "$pip_reqs" >> tmp.txt
    done
    mv tmp.txt "$pip_reqs"

    install_gradle=true
    rm_pkgs="$rm_pkgs unzip wget"
fi

echo -e "\nUpdating package index..."
apt-get update

if [ -n "$apt_pkgs" ]
then
    # Install required apt packages
    echo -e "\nInstalling apt packages: $apt_pkgs"
    apt-get install -y --no-install-recommends $apt_pkgs
fi

if [ -f "$pip_reqs" ]
then
    # Install required pip packages
    python3 --version
    pip3 --version
    echo -e "\nInstalling pip packages: `cat "$pip_reqs" | tr '\n' ' '`" 
    pip3 install --no-cache-dir --disable-pip-version-check -r "$pip_reqs"
fi

if [ "$install_gradle" = true ]
then
    javac --version
    echo -e "\nInstalling gradle"
    wget --no-verbose https://services.gradle.org/distributions/${GRADLE_VERSION}-bin.zip
    unzip ${GRADLE_VERSION}-bin.zip -d /opt
    rm ${GRADLE_VERSION}-bin.zip
    gradle --version
fi

if [ -n "$rm_pkgs" ]
then
    # Remove packages needed only for install
    echo -e "\nRemoving packages only required for install: $rm_pkgs"
    apt-get purge -y --autoremove $rm_pkgs
fi

rm -rf /var/lib/apt/lists/*

echo ""
echo "-=------------------------------------------------------"
echo " Installation complete for service '$service'!"
echo "-=------------------------------------------------------"
echo ""

