
# CHANGELOG
All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). See the [CONTRIBUTING guide](./CONTRIBUTING.md#Changelog) for instructions on how to add changelog entries.

## [Unreleased 3.6](https://github.com/opensearch-project/opensearch-jvector/compare/3.5...HEAD)
### Features
### Enhancements
### Bug Fixes
- Fix derived nested fields processing [484] (https://github.com/opensearch-project/opensearch-jvector/pull/484)
### Infrastructure
* Upgrade Lucene to 10.4.0 [292] (https://github.com/opensearch-project/opensearch-jvector/pull/292)
* Upgrade jvector from 4.0.0-rc.6 to 4.0.0-rc.8 [370](https://github.com/opensearch-project/opensearch-jvector/pull/370)
* Update Gradle to 9.4.1 [381](https://github.com/opensearch-project/opensearch-jvector/pull/381)
* Update Sptoless to 8.4.0 [495](https://github.com/opensearch-project/opensearch-jvector/pull/495)
### Documentation
### Documentation
* Update user guide covering index creation, search tuning, compression levels, and advanced topics; add demo script walking through cluster health check, bulk indexing, filtered/tuned KNN search, force merge, and node stats
### Maintenance
* Fix String.format() uses the default system locale [465](https://github.com/opensearch-project/opensearch-jvector/pull/465)
### Refactoring
