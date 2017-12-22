SHELL := /bin/bash
VE_ROOT := /tmp

all: clean install run
	echo "Running all."

run:
	VE_ROOT=$(VE_ROOT) bash ./course_assignments/run.sh

install: install_python_env
	$(MAKE) -C ./virtual_environment install

install_python_env:
	sudo apt-get install python-numpy python-scipy python-matplotlib

clean:
	rm -rf $(VE_ROOT)/google_dl
