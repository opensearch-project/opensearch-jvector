
# CHANGELOG
All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). See the [CONTRIBUTING guide](./CONTRIBUTING.md#Changelog) for instructions on how to add changelog entries.

## [Unreleased 3.3](https://github.com/opensearch-project/opensearch-jvector/compare/3.3...HEAD)
### Features
### Enhancements
* More tests for leading segment merge. [PR #243](https://github.com/opensearch-project/opensearch-jvector/pull/243)
### Bug Fixes
* Revert of support deletes for incremental insertion [240](https://github.com/opensearch-project/opensearch-jvector/pull/240)
* Fix visited docs tracking that led to assertions on AbstractKnnVectorQuery side [238] (https://github.com/opensearch-project/opensearch-jvector/pull/238)
* Fix leading segment merge for deletions [242](https://github.com/opensearch-project/opensearch-jvector/pull/242)
### Infrastructure
* Update OpenSearch compatibility from version 3.3.0 to 3.3.2 [PR #226](https://github.com/opensearch-project/opensearch-jvector/pull/226)
### Documentation
### Maintenance
* Fix vulnerability due to matplotlib CVE-66034 [223] (https://github.com/opensearch-project/opensearch-jvector/pull/223)
### Refactoring
