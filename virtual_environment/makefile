SHELL := /bin/bash
VE_ROOT := /tmp

install: install_virtual_environment install_pip_packages
	:

install_virtual_environment:
	sudo pip3 install virtualenv
	virtualenv $(VE_ROOT)/google_dl

install_pip_packages:
	# make[1]: source: Command not found (ubuntu uses dash, not sh in /bin/sh)...
	# source $(VE_ROOT)/google_dl/bin/activate && which python3 && which pip3 && pip3 install -r ./pip3_environment.txt
	source $(VE_ROOT)/google_dl/bin/activate && (cat ./pip3_environment.txt | sed -E "s/=.*//g" | xargs -l pip3 install || : )
