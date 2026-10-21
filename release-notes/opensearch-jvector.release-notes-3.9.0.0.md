## Version 3.9.0.0 Release Notes

Compatible with OpenSearch and OpenSearch Dashboards version 3.9.0

### Enhancements
- [Storage perf] Stop writing the redundant binary doc values copy of the vector for jVector fields from 3.9.0 onwards. Preserve existing indices. [715] (https://github.com/opensearch-project/opensearch-jvector/pull/715) 
- Remove training run measuments since refining was removed [734] (https://github.com/opensearch-project/opensearch-jvector/pull/734) 

### Bug Fixes
- Fix flaky testJVectorKnnIndex_filter_maxInnerProduct test case [681](https://github.com/opensearch-project/opensearch-jvector/pull/681)
- Fix flaky testJVectorKnnIndex_simpleCase  test case [692](https://github.com/opensearch-project/opensearch-jvector/pull/692)

### Documentation
- Update release procedure [690](https://github.com/opensearch-project/opensearch-jvector/pull/690)

### Maintenance
- Remove deprecated use_pruning features [670](https://github.com/opensearch-project/opensearch-jvector/pull/670)