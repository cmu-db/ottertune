from fabric.api import local

PATH = "/var/www/ottertune"
USER = "www-data"
local("sudo chown -R {0}:{0} {1}".format(USER, PATH))
local("sudo chmod -R ugo+rX,ug+w {}".format(PATH))
