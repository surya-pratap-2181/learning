SHELL := /bin/bash
RUBY_BIN := /opt/homebrew/opt/ruby/bin
GEM_BIN := /opt/homebrew/lib/ruby/gems/4.0.0/bin
BUNDLE := PATH=$(RUBY_BIN):$(GEM_BIN):$$PATH bundle

# Filter out Sass deprecation warnings from the theme (not our code)
FILTER := 2> >(grep -v -E 'Deprecation Warning|deprecated|sass-lang.com/d/|More info and automated migrator|color\.(scale|adjust)|map\.get instead|──|╷|│|╵|\.scss [0-9]|@import$$|root stylesheet$$|repetitive deprecation|Run in verbose mode' >&2)

.PHONY: install serve build stop clean help

## install: Install Ruby dependencies
install:
	$(BUNDLE) config set path 'vendor/bundle'
	$(BUNDLE) install

## serve: Start local dev server with live reload
serve:
	$(BUNDLE) exec jekyll serve --livereload $(FILTER)

## build: Build the site to _site/
build:
	$(BUNDLE) exec jekyll build $(FILTER)

## stop: Stop running Jekyll server
stop:
	pkill -f jekyll || true

## clean: Remove generated files
clean:
	rm -rf _site .jekyll-cache .sass-cache

## help: Show available commands
help:
	@grep -E '^## ' Makefile | sed 's/## //' | column -t -s ':'
