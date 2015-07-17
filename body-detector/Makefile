all:
	mkdir -p tmp
	mkdir -p lib
	touch lib/__init__.py
	python setup.py build_ext --build-temp tmp --build-lib lib --pyrex-c-in-temp

install:
	python setup.py install

pdf:
	enscript  --color -E -q -Z -p - -f Courier10 train.py | ps2pdf - train.pdf
	enscript  --color -E -q -Z -p - -f Courier10 predict.py | ps2pdf - predict.pdf

clean:
	rm lib/*.so
	rm -r tmp
