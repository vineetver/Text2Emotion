SHELL = /bin/bash


.PHONY: help
help:
	@echo "Usage: make [target]"

	@echo "Available targets:""
	@echo "venv        : Create a virtualenv"
	@echo "clean       : Remove build artifacts"
	@echo "test        : Execute unit and integration tests"


.ONESHELL:
venv:
	@echo "Creating virtualenv"
	python3 -m venv venv && source venv/bin/activate
	# if cudnn is not installed, install it
	if nvcc --version | grep -q "not found"; then
		@echo "CUDNN not found, installing CUDNN"
		# if not on ubuntu 20.04 raise
		if ! lsb_release -a | grep -q "Ubuntu 20.04"; then
			@echo "This script is only available on Ubuntu 20.04. Please install CUDNN manually for your system."
			exit 1
		fi
		wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcudnn8_8.1.1.33-1+cuda11.2_amd64.deb
		https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcudnn8-dev_8.1.1.33-1+cuda11.2_amd64.deb
		sudo dpkg -i libcudnn8_8.1.1.33-1+cuda11.2_amd64.deb
		sudo dpkg -i libcudnn8-dev_8.1.1.33-1+cuda11.2_amd64.deb
		sudo apt-get install -f  # re
	fi
	pip install -r requirements.txt
	@echo "Virtualenv successfully created"

.PHONY: clean
clean:
	@echo "Cleaning build artifacts"
	rm -rf stores
	rm -rf logs
	@echo "Build artifacts successfully removed"

.PHONY: test
test:
	@echo "Executing unit tests"
	pytest -m "not integration"
	cd tests && great_expectations checkpoint run ___
	cd tests && great_expectations checkpoint run ___
	cd tests && great_expectations checkpoint run ___

	@echo "Executing integration tests"
	pytest -m "integration"
	cd tests && great_expectations checkpoint run ___
	cd tests && great_expectations checkpoint run ___
	cd tests && great_expectations checkpoint run ___
