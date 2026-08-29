
# CHANGELOG
All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). See the [CONTRIBUTING guide](./CONTRIBUTING.md#Changelog) for instructions on how to add changelog entries.

## [Unreleased 3.8](https://github.com/opensearch-project/opensearch-jvector/compare/3.8...HEAD)
### Features

### Enhancements
- [Storage perf] Stop writing the redundant binary doc values copy of the vector for jVector fields from 3.9.0 onwards. Preserve existing indices. 

### Bug Fixes
- Fix flaky testJVectorKnnIndex_filter_maxInnerProduct test case [681](https://github.com/opensearch-project/opensearch-jvector/pull/681)
- Fix flaky testJVectorKnnIndex_simpleCase  test case [692](https://github.com/opensearch-project/opensearch-jvector/pull/692)

### Infrastructure
### Documentation
- Update release procedure [690](https://github.com/opensearch-project/opensearch-jvector/pull/690)
### Maintenance
- Remove deprecated use_pruning features [670](https://github.com/opensearch-project/opensearch-jvector/pull/670)
### Refactoring
