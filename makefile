SHELL := /bin/bash
VE_ROOT := /tmp
PY36_ROOT := /tmp/py36
PY36_PREFIX := /tmp/config_prefix

all: clean install run
	echo "Running all."

run:
	VE_ROOT=$(VE_ROOT) bash ./course_assignments/run.sh

install: install_python36 install_python_env
	$(MAKE) -C ./virtual_environment install

install_python36: $(PY36_PREFIX)/bin/python3.6
	:

$(PY36_PREFIX)/bin/python3.6:
	# Install requirements.
	# sudo apt-get install build-essential
	# sudo apt-get install libssl-dev zlib1g-dev libncurses5-dev libncursesw5-dev libreadline-dev libsqlite3-dev libgdbm-dev libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev tk-dev
	mkdir $(PY36_ROOT) $(PY36_PREFIX)
	cd $(PY36_ROOT) && wget -O - www.python.org/ftp/python/3.6.4/Python-3.6.4.tar.xz | tar -xJf -
	# In ./configure the '--enable-optimizations' flags may be useful but it increases compilation time.
	cd $(PY36_ROOT)/Python-3.6.4/ && ./configure --prefix=$(PY36_PREFIX) --enable-optimizations
	cd $(PY36_ROOT)/Python-3.6.4/ && make -j 3
	cd $(PY36_ROOT)/Python-3.6.4/ && make altinstall

install_python_env:
	# sudo apt-get install python-numpy python-scipy python-matplotlib

clean:
	rm -rf $(PY36_ROOT)
	rm -rf $(PY36_PREFIX)
	rm -rf $(VE_ROOT)/google_dl
