SHELL := /bin/bash
VE_ROOT := /tmp
PY36_ROOT := /tmp/py36
PY36_PREFIX := /tmp/config_prefix

install: install_virtual_environment install_pip_packages
	:

install_virtual_environment: .FORCE
	sudo pip3 install virtualenv
	virtualenv --system-site-packages --python=$(PY36_PREFIX)/bin/python3.6 $(VE_ROOT)/google_dl

install_pip_packages: .FORCE
	# make[1]: source: Command not found (ubuntu uses dash, not sh in /bin/sh)...
	# source $(VE_ROOT)/google_dl/bin/activate && which python3 && which pip3 && pip3 install -r ./pip3_environment.txt
	source $(VE_ROOT)/google_dl/bin/activate && (cat ./pip3_environment.txt | sed -E "s/=.*//g" | xargs -l pip3 install || : )
	sudo pip3 install --upgrade pip setuptools
	sudo pip3 install virtualenv
	virtualenv google_dl

.FORCE:
