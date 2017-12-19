install: install_virtual_environment install_pip_packages
	:

install_virtual_environment:
	pip3 install virtualenv
	virtualenv google_dl

install_pip_packages:
	source ./google_dl/bin/activate
	sudo pip3 install ./pip3_environment.txt
	pip3 freeze
