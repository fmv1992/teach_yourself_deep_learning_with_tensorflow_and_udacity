all: install run
	echo "Running all."

run:
	bash ./course_assignments/run.sh

install: install_python_env
	$(MAKE) -C ./virtual_environment install

install_python_env:
	sudo apt-get install python-numpy python-scipy python-matplotlib
