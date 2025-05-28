
# CHANGELOG
All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). See the [CONTRIBUTING guide](./CONTRIBUTING.md#Changelog) for instructions on how to add changelog entries.

## [Unreleased 3.0](https://github.com/opensearch-project/k-NN/compare/2.x...HEAD)
### Features
* Add jVector search query statistics [#62](https://github.com/opensearch-project/opensearch-jvector/issues/62)
### Enhancements
* Upgrade to java 22 so that we can use Foreign Memory API and MemorySegmentReader
* Clone instead of recreating index inputs [#83](https://github.com/opensearch-project/opensearch-jvector/issues/83)
* Switch to using Lucene indexInput directly and fix integrity verification for vector index files
* Make merges happen largely off heap [#87](https://github.com/opensearch-project/opensearch-jvector/issues/87)
### Bug Fixes
* Fix the limitation of the 2 GB segment with backwards compatibility to JDK21 and move default build to 21 target
* Fix file handle leak during queries
### Infrastructure
### Documentation
### Maintenance
### Refactoring
Refactor to use the utility method of merged values in writer and add deleted doc tests [#79](https://github.com/opensearch-project/opensearch-jvector/issues/79)

## [Unreleased 2.x](https://github.com/opensearch-project/k-NN/compare/2.18...2.x)
### Features
### Enhancements
### Bug Fixes
### Infrastructure
### Documentation
### Maintenance
### Refactoring
