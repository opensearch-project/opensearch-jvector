## Version 3.6.0.0 Release Notes

Compatible with OpenSearch and OpenSearch Dashboards version 3.6.0

### Features
* Implemented Maximal Marginal Relevance (MMR) search feature for diversified vector search results [253] (https://github.com/opensearch-project/opensearch-jvector/issues/253)
* Enable derived sources for vectors to save storage costs. [241] (https://github.com/opensearch-project/opensearch-jvector/pull/241)
### Enhancements
### Bug Fixes
* Fix issues handling documents without vector fields being populated [288] (https://github.com/opensearch-project/opensearch-jvector/issues/288)
* The KNN1030Codec does not properly support delegation for non-default codec(s). [310] (https://github.com/opensearch-project/opensearch-jvector/pull/310)
* Remove usage of the commons-lang 2.6 [317] (https://github.com/opensearch-project/opensearch-jvector/pull/317)
* Fixing guava dependency scope, since the dependency is provided by transport-grpc plugin [344] (https://github.com/opensearch-project/opensearch-jvector/pull/344)
* Remove manual ref counting and simplify DeriveSourceReaders [397] (https://github.com/opensearch-project/opensearch-jvector/pull/397)
* Fix CVE-2026-28684: Upgrade python-dotenv to 1.2.2 to address symbolic link following vulnerability [448] (https://github.com/opensearch-project/opensearch-jvector/issues/448)
### Infrastructure
* Upgrade Lucene to 10.4.0 [292] (https://github.com/opensearch-project/opensearch-jvector/pull/292)
### Documentation
### Maintenance
* Update `com.google.guava:failureaccess` from 1.0.1 to 1.0.2
* Update `com.google.guava:guava` from 32.1.3-jre to 33.2.1-jre
* Move legacy codecs to backward_codecs [389](https://github.com/opensearch-project/opensearch-jvector/pull/389)
* Remove 9.x codecs [411](https://github.com/opensearch-project/opensearch-jvector/pull/411)
### Refactoring

