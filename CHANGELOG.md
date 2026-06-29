
# CHANGELOG
All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). See the [CONTRIBUTING guide](./CONTRIBUTING.md#Changelog) for instructions on how to add changelog entries.

## [Unreleased 3.7](https://github.com/opensearch-project/opensearch-jvector/compare/3.7...HEAD)
### Features
- Added capability to retrieve float, binary and byte data types vectors using doc_values [#3321](https://github.com/opensearch-project/k-NN/pull/3321)

### Enhancements
### Bug Fixes
- Fix dynamic template and mixed cases [538](https://github.com/opensearch-project/opensearch-jvector/pull/538)
- Fix flaky testMixedBatchSizesForQuantization test case [564](https://github.com/opensearch-project/opensearch-jvector/pull/564)
- Fix flaky testJVectorKnnIndex_simpleCase_maxInnerProduct test case [569](https://github.com/opensearch-project/opensearch-jvector/pull/569)
- Fix NPE due to an attempt to read the segment with no graph [552](https://github.com/opensearch-project/opensearch-jvector/pull/552)

### Infrastructure
### Documentation
### Maintenance
### Refactoring
