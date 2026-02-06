
# CHANGELOG
All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). See the [CONTRIBUTING guide](./CONTRIBUTING.md#Changelog) for instructions on how to add changelog entries.

## [Unreleased 3.4](https://github.com/opensearch-project/opensearch-jvector/compare/3.3...HEAD)
### Features
### Enhancements
* More tests for leading segment merge. [243] (https://github.com/opensearch-project/opensearch-jvector/pull/243)
* Include AdditionalCodecs argument to allow additional Codec registration [250] (https://github.com/opensearch-project/opensearch-jvector/pull/250)

### Bug Fixes
* Fix visited docs tracking that led to assertions on AbstractKnnVectorQuery side [238] (https://github.com/opensearch-project/opensearch-jvector/pull/238)
* Revert of support deletes for incremental insertion [240](https://github.com/opensearch-project/opensearch-jvector/pull/240)
* Fix leading segment merge for deletions [242](https://github.com/opensearch-project/opensearch-jvector/pull/242)
### Infrastructure
* Upgrade Gradle to 9.2.0 [220] (https://github.com/opensearch-project/opensearch-jvector/pull/222)
* Add support for JDK25 [220] (https://github.com/opensearch-project/opensearch-jvector/pull/222)
### Documentation
### Maintenance
* Fix vulnerability due to matplotlib CVE-66034 [223] (https://github.com/opensearch-project/opensearch-jvector/pull/223)
* Disable some Github workflows since those are failing consistently [231] (https://github.com/opensearch-project/opensearch-jvector/pull/231)
* Fix BWC Github Actions, enable all Github workflows [234] (https://github.com/opensearch-project/opensearch-jvector/pull/234)
### Refactoring
