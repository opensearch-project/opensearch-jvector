
# CHANGELOG
All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). See the [CONTRIBUTING guide](./CONTRIBUTING.md#Changelog) for instructions on how to add changelog entries.

## [Unreleased 3.6](https://github.com/opensearch-project/opensearch-jvector/compare/3.5...HEAD)
### Features
* Implemented Maximal Marginal Relevance (MMR) search feature for diversified vector search results [253] (https://github.com/opensearch-project/opensearch-jvector/issues/253)
* Enable derived sources for vectors to save storage costs. [241] (https://github.com/opensearch-project/opensearch-jvector/pull/241)
### Enhancements
### Bug Fixes
* Fix issues handling documents without vector fields being populated [288] (https://github.com/opensearch-project/opensearch-jvector/issues/288)
* The KNN1030Codec does not properly support delegation for non-default codec(s). [310] (https://github.com/opensearch-project/opensearch-jvector/pull/310)
* Remove usage of the commons-lang 2.6 [317] (https://github.com/opensearch-project/opensearch-jvector/pull/317)
### Infrastructure
* Upgrade Lucene to 10.4.0 [292] (https://github.com/opensearch-project/opensearch-jvector/pull/292)
### Documentation
### Maintenance
### Refactoring
