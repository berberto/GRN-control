PIP ?= pip

all: clean install

clean:
	rm -rf *.egg-info
	rm -f `find . -type f -name \*.py[co]`

install:
	$(PIP) install -e .
