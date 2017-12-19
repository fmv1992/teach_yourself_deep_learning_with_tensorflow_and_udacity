SHELL := /bin/bash

install: install_virtual_environment install_pip_packages
	:

install_virtual_environment:
	sudo pip3 install virtualenv
	virtualenv google_dl

install_pip_packages:
	# make[1]: source: Command not found (ubuntu uses dash, not sh in /bin/sh)...
	source ./google_dl/bin/activate
	pip3 install -r ./pip3_environment.txt
	pip3 freeze
