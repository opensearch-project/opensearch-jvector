## Version 3.10.0.0 Release Notes

Compatible with OpenSearch and OpenSearch Dashboards version 3.10.0

### Features
- Added abort checking for graph constructions [761](https://github.com/opensearch-project/opensearch-jvector/pull/761)

### Enhancements
- Incremental insertion to leading segment for PQ quantized graphs [728](https://github.com/opensearch-project/opensearch-jvector/pull/728)
- Update datastax/jvector library to 4.0.1 [745](https://github.com/opensearch-project/opensearch-jvector/pull/745)
- Use FloatArray to eliminate unnecessary vector copying if possible [747](https://github.com/opensearch-project/opensearch-jvector/pull/747)
- Update datastax/jvector library to 4.0.2 
- Warm OS page cache during segment merge via jvector prefetchL0Records [790](https://github.com/opensearch-project/opensearch-jvector/pull/790)
- Add KNNQueryBuilder::size property [797](https://github.com/opensearch-project/opensearch-jvector/pull/797)

### Bug Fixes
- Excessive heap consumption when using PQ/NVQ vectors with derived sources enabled [754](https://github.com/opensearch-project/opensearch-jvector/pull/754)
- Fix RNNQueryFactory issue after lucene upgrade to 10.5.2 [769](https://github.com/opensearch-project/opensearch-jvector/pull/769)
- Allows k to be nullable to fix filter bug [2836](https://github.com/opensearch-project/k-NN/issues/2836)
- Wrap Lucene's KnnXxx queries into LuceneEngineKnnVectorQuery (to address neural-search plugin compatibility) [817](https://github.com/opensearch-project/k-NN/issues/817)

### Documentation
- Point the CI and coverage badges in the README at this repository [752](https://github.com/opensearch-project/opensearch-jvector/pull/752)

### Maintenance
- Removed unused public classes, interfaces and enums related with Quantization state and cache [764](https://github.com/opensearch-project/opensearch-jvector/pull/764)
* Fix vulnerability due to matplotlib [810] (https://github.com/opensearch-project/opensearch-jvector/pull/810)
* Fix vulnerability due to pyarrow [811] (https://github.com/opensearch-project/opensearch-jvector/pull/811)