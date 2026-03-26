## Version 3.3.2.2 Release Notes

Compatible with OpenSearch and OpenSearch Dashboards version 3.3.2

### Features
### Enhancements
* More tests for leading segment merge. [PR #243](https://github.com/opensearch-project/opensearch-jvector/pull/243)
### Bug Fixes
* Revert of support deletes for incremental insertion [240](https://github.com/opensearch-project/opensearch-jvector/pull/240)
* Fix visited docs tracking that led to assertions on AbstractKnnVectorQuery side [238] (https://github.com/opensearch-project/opensearch-jvector/pull/238)
* Fix leading segment merge for deletions [242](https://github.com/opensearch-project/opensearch-jvector/pull/242)
### Infrastructure
* Update OpenSearch compatibility from version 3.3.0 to 3.3.2 [PR #226](https://github.com/opensearch-project/opensearch-jvector/pull/226)
### Documentation
### Maintenance
* Fix vulnerability due to matplotlib CVE-66034 [223] (https://github.com/opensearch-project/opensearch-jvector/pull/223)
### Refactoring
