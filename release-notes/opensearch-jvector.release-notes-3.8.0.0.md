## Version 3.7.0.0 Release Notes

Compatible with OpenSearch and OpenSearch Dashboards version 3.8.0

### Features
- Add NVQ Quantization [539](https://github.com/opensearch-project/opensearch-jvector/pull/539) 
- Added capability to retrieve float, binary and byte data types vectors using doc_values [538](https://github.com/opensearch-project/opensearch-jvector/pull/538)
- Enable native vectorization provider [622](https://github.com/opensearch-project/opensearch-jvector/pull/622)

### Enhancements
- Introduce extensible VectorSearchEngine API [650](https://github.com/opensearch-project/opensearch-jvector/pull/650)

### Bug Fixes
- Fix dynamic template and mixed cases [538](https://github.com/opensearch-project/opensearch-jvector/pull/538)
- Fix flaky testMixedBatchSizesForQuantization test case [564](https://github.com/opensearch-project/opensearch-jvector/pull/564)
- Fix flaky testJVectorKnnIndex_simpleCase_maxInnerProduct test case [569](https://github.com/opensearch-project/opensearch-jvector/pull/569)
- Preserve non-XContent `_source` fields when derived source is enabled [624] (https://github.com/opensearch-project/opensearch-jvector/pull/624)
- Remove PQ codebook refinement due to diminishing value [662] (https://github.com/opensearch-project/opensearch-jvector/pull/662)