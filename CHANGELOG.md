
# CHANGELOG
All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). See the [CONTRIBUTING guide](./CONTRIBUTING.md#Changelog) for instructions on how to add changelog entries.

## [Unreleased 3.7](https://github.com/opensearch-project/opensearch-jvector/compare/3.7...HEAD)
### Features
- Add NVQ Quantization [#539](https://github.com/opensearch-project/opensearch-jvector/pull/539) 
- Added capability to retrieve float, binary and byte data types vectors using doc_values [#538](https://github.com/opensearch-project/opensearch-jvector/pull/538)

### Enhancements
### Bug Fixes
- Fix dynamic template and mixed cases [538](https://github.com/opensearch-project/opensearch-jvector/pull/538)
- Fix flaky testMixedBatchSizesForQuantization test case [564](https://github.com/opensearch-project/opensearch-jvector/pull/564)
- Fix flaky testJVectorKnnIndex_simpleCase_maxInnerProduct test case [569](https://github.com/opensearch-project/opensearch-jvector/pull/569)
- Preserve non-XContent `_source` fields when derived source is enabled [624] (https://github.com/opensearch-project/opensearch-jvector/pull/624)

### Infrastructure
### Documentation
### Maintenance
### Refactoring
