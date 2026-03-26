## Version 3.4.0.0 Release Notes

Compatible with OpenSearch and OpenSearch Dashboards version 3.4.0

### Features
### Enhancements
* More tests for leading segment merge. [PR #243](https://github.com/opensearch-project/opensearch-jvector/pull/243)
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
### Refactoring
