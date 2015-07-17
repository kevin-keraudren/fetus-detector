TMP_DIR := tmp
SHELL := /bin/bash

all:
	mkdir -p ${TMP_DIR}
	python setup.py build_ext --build-temp ${TMP_DIR} --build-lib lib/ --pyrex-c-in-temp #--force

