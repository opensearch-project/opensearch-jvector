- [Overview](#overview)
- [Branching](#branching)
  - [Release Branching](#release-branching)
  - [Feature Branches](#feature-branches)
- [Release Labels](#release-labels)
- [Releasing](#releasing)

## Overview

This document explains the release strategy for artifacts in this organization.

## Branching

### Release Branching

Given the current major release of 1.0, projects in this organization maintain the following active branches.

* **main**: The next _major_ release. This is the branch where all merges take place and code moves fast.
* **1.x**: The next _minor_ release. Once a change is merged into `main`, decide whether to backport it to `1.x`.
* **1.0**: The _current_ release. In between minor releases, only hotfixes (e.g. security) are backported to `1.0`.

Label PRs with the next major version label (e.g. `2.0.0`) and merge changes into `main`. Label PRs that you believe need to be backported as `1.x` and `1.0`. Backport PRs by checking out the versioned branch, cherry-pick changes and open a PR against each target backport branch.

### Feature Branches

Do not creating branches in the upstream repo, use your fork, for the exception of long lasting feature branches that require active collaboration from multiple developers. Name feature branches `feature/<thing>`. Once the work is merged to `main`, please make sure to delete the feature branch.

## Release Labels

Repositories create consistent release labels, such as `v1.0.0`, `v1.1.0` and `v2.0.0`, as well as `backport`. Use release labels to target an issue or a PR for a given release. See [MAINTAINERS](MAINTAINERS.md#triage-open-issues) for more information on triaging issues.

## Backwards Compatibility

[The backwards compatibility test suite](qa) is used to ensure upgrades to the current version are successful. 
When releasing a new version, update the `bwc.version` to the latest, previous minor version in [gradle.properties](gradle.properties). 

## Releasing

The release process is standard across repositories in this org and is run by a release manager volunteering from amongst [MAINTAINERS](MAINTAINERS.md).

### Standalone Maven Central Release (Before Onboarding Bundle)

**DO NOT cut a tag by going to release section of Github UI. It will mess up the Github Action.**

Note: A maintainer must remember to perform steps 1, 2 and 4 (require total of 3 maintainers, 1 cut tag, 2 approve).
1. Run these commands from the upstream opensearch-jvector repository, not a forked one: 
```
git checkout main
git fetch origin
git rebase origin/main
git tag <version>
git push origin <version> 
```
2. Wait for Github Actions to run and open the newly created issue. Two maintainers should comment `approve` in the issue.
3. Wait for Jenkins to be triggered, pull the artifacts built by Actions, push to sonatype release channel on remote. Wait for an hour or so for Sonatype to copy it into Maven Central.
4. Bump [build.gradle](./build.gradle), update [release-notes](./release-notes/), and clean up entries from [CHANGELOG.md](./CHANGELOG.md) via a PR.


Thanks.

