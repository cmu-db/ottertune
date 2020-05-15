#!/bin/bash

# Tells bash to exit if any command returns a non-zero return value
set -e

service="$1"

echo ""
echo "-=------------------------------------------------------"
echo " Starting installation for service '$service'..."
echo "-=------------------------------------------------------"

if [ "$DEBUG" = true ]
then
    echo ""
    echo "Command-line Args:"
    echo " - service: $service"
    echo ""
    echo "Environment Variables:"
    echo " - DEBIAN_FRONTEND: $DEBIAN_FRONTEND"
    echo " - PATH:            $PATH"
    echo " - GRADLE_VERSION:  $GRADLE_VERSION"
    echo " - GRADLE_HOME:     $GRADLE_HOME"
    echo ""
fi

apt_pkgs=""
rm_pkgs=""
install_gradle=false
pip_common_pkgs="Fabric3 numpy requests"
master_pip_reqs_file=/requirements.txt
pip_reqs_file="/${service}-requirements.txt"

if [ "$service" = "base" ]
then
    apt_pkgs="python3.6 python3-setuptools python3-pip libssl-dev vim dnsutils iputils-ping"

    # Filter common pip packages
    for pip_pkg in $pip_common_pkgs
    do
        grep "^$pip_pkg" "$master_pip_reqs_file" >> "$pip_reqs_file"
    done

elif [ "$service" = "web" ]
then
    apt_pkgs="python3-dev gcc g++ mysql-client libmysqlclient-dev python-mysqldb postgresql-client telnet"

    rm_pkgs="gcc g++"

    pip_skip_pkgs="$pip_common_pkgs astroid autopep8 git-lint pycodestyle pylint"
    cp "$master_pip_reqs_file" "$pip_reqs_file"

    for pip_pkg in $pip_skip_pkgs
    do
        sed -i "/$pip_pkg/d" "$pip_reqs_file"
    done

elif [ "$service" = "driver" ]
then
    apt_pkgs="openssh-server openjdk-11-jdk unzip wget"
    rm_pkgs="unzip wget"
    install_gradle=true

else
    echo ""
    echo "ERROR: Invalid value for service: '$service'"
    echo ""
    echo "Usage: $0 [base|web|driver]"
    exit 1
fi

echo -e "\nUpdating package index..."
apt-get update

if [ -n "$apt_pkgs" ]
then
    # Install required apt packages
    echo -e "\nInstalling apt packages: $apt_pkgs"
    apt-get install -y --no-install-recommends $apt_pkgs
fi

if [ -f "$pip_reqs_file" ] && [ -s "$pip_reqs_file" ]
then
    # Install required pip packages
    python3 --version
    pip3 --version
    echo -e "\nInstalling pip packages: `cat "$pip_reqs_file" | tr '\n' ' '`"
    pip3 install --no-cache-dir --disable-pip-version-check -r "$pip_reqs_file"
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

