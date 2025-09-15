SHELL := /bin/bash

.PHONY: install start build test lint clean

install:
	npm install

start:
	npm start

build:
	npm run build

test:
	npm test

lint:
	npx eslint . || true

clean:
	rm -rf node_modules build
