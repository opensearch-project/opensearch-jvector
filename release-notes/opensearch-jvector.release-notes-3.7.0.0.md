## Version 3.7.0.0 Release Notes

Compatible with OpenSearch and OpenSearch Dashboards version 3.7.0

### Features
* Add gRPC Protocol Buffer support for KNN queries [391](https://github.com/opensearch-project/opensearch-jvector/issues/391)
* Support derived source for knn with other fields [474] (https://github.com/opensearch-project/opensearch-jvector/pull/488)
* Add support for MAXIMUM_INNER_PRODUCT similarity function in jVector engine ([#206](https://github.com/opensearch-project/opensearch-jvector/pull/494))
### Enhancements
### Bug Fixes
- Fix derived nested fields processing [484] (https://github.com/opensearch-project/opensearch-jvector/pull/484)
- Fix GRPC integration flaky test caused by locale-sensitive String.format [#505](https://github.com/opensearch-project/opensearch-jvector/pull/505)
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

