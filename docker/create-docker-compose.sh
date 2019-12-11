#!/bin/bash


if [ -z "$BACKEND" ]
then
    echo "Variable 'BACKEND' must be set."
    exit 1
fi

DEBUG="${DEBUG:-true}"
ADMIN_PASSWORD="${ADMIN_PASSWORD:-changeme}"
DB_NAME="${DB_NAME:-ottertune}"
DB_PASSWORD="${DB_PASSWORD:-ottertune}"

if [ "$BACKEND" = "mysql" ]; then
    DB_USER="${DB_USER:-root}"
    DB_PORT="${DB_PORT:-3306}"
else
    DB_USER="${DB_USER:-postgres}"
    DB_PORT="${DB_PORT:-5432}"
fi

WEB_ENTRYPOINT="${WEB_ENTRYPOINT:-./start.sh}"

file="$(test -z "$1" && echo "docker-compose.$BACKEND.yml" || echo "$1")"


cat > $file <<- EOM
version: "3"
services:

    web:
        image: ottertune-web
        container_name: web
        expose:
          - "8000"
        ports:
          - "8000:8000"
        links:
          - backend
          - rabbitmq
        depends_on:
          - backend
          - rabbitmq
        environment:
          DEBUG: '$DEBUG'
          ADMIN_PASSWORD: '$ADMIN_PASSWORD'
          BACKEND: '$BACKEND'
          DB_NAME: '$DB_NAME'
          DB_USER: '$DB_USER'
          DB_PASSWORD: '$DB_PASSWORD'
          DB_HOST: 'backend'
          DB_PORT: '$DB_PORT'
          MAX_DB_CONN_ATTEMPTS: 30
          RABBITMQ_HOST: 'rabbitmq'
        working_dir: /app/website
        entrypoint: $WEB_ENTRYPOINT
        labels:
          NAME: "ottertune-web"
        networks:
          - ottertune-net

    driver:
        image: ottertune-driver
        container_name: driver
        depends_on:
          - web
        environment:
          DEBUG: '$DEBUG'
        working_dir: /app/driver
        labels:
          NAME: "ottertune-driver"
        networks:
          - ottertune-net

    rabbitmq:
        image: "rabbitmq:3-management"
        container_name: rabbitmq
        restart: always
        hostname: "rabbitmq"
        environment:
           RABBITMQ_DEFAULT_USER: "guest"
           RABBITMQ_DEFAULT_PASS: "guest"
           RABBITMQ_DEFAULT_VHOST: "/"
        expose:
           - "15672"
           - "5672"
        ports:
           - "15673:15672"
           - "5673:5672"
        labels:
           NAME: "rabbitmq"
        networks:
          - ottertune-net

EOM


cat >> $file <<- EOM
    backend:
        container_name: backend
        restart: always
EOM


if [ "$BACKEND" = "mysql" ]; then

cat >> $file <<- EOM
        image: mysql:5.7
        environment:
          MYSQL_USER: '$DB_USER'
          MYSQL_ROOT_PASSWORD: '$DB_PASSWORD'
          MYSQL_PASSWORD: '$DB_PASSWORD'
          MYSQL_DATABASE: '$DB_NAME'
        expose:
          - "3306"
        ports:
          - "3306:3306"
EOM
else
cat >> $file <<- EOM
        image: postgres:9.6
        environment:
          POSTGRES_PASSWORD: '$DB_PASSWORD'
          POSTGRES_USER: '$DB_USER'
          POSTGRES_DB: '$DB_NAME'
        expose:
          - "5432"
        ports:
          - "5432:5432"
EOM
fi

cat >> $file <<- EOM
        labels:
          NAME: "ottertune-backend"
        networks:
          - ottertune-net

networks:
   ottertune-net:
      driver: bridge
EOM

echo "Saved docker-compose file to '$file'."

