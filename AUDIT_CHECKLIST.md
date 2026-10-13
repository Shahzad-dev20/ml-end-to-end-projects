# Repository Audit Checklist

## Critical

- [x] Confirm repository scope, branch, remote, and tracked files
- [x] Verify there are no active secrets or credential files in the working tree
- [x] Identify and document the untracked fish-weight-body-fat-predictor folder before any public push
- [x] Remove the hardcoded personal dataset path from the EXIM notebook configuration

## High

- [x] Rewrite the root README so it accurately reflects the current repository structure
- [x] Correct incorrect House Price project run instructions
- [x] Improve repository-level hygiene for local environment and cache files
- [x] Confirm project folders are self-contained and their README expectations match the repository contents

## Medium

- [x] Review README accuracy across all project folders
- [x] Check for duplicate or stale artifacts that may confuse GitHub viewers
- [x] Identify project naming inconsistencies such as spaces in folder names and mixed casing
- [x] Document the fact that datasets are intentionally external for most projects

## Low

- [x] Preserve existing project work without unnecessary reorganization or destructive cleanup
- [x] Keep notebook and script behavior compatible with the current project folders
- [ ] Decide whether to include the fish project in the first official GitHub push
- [ ] Add project-level contribution or licensing files only if the owner wants a formal repo policy

## Summary of the audit

The repository is structurally sound for a portfolio GitHub push, but it needs a careful review before the first public release. The main corrections were documentation drift, a stale local path, and a missing Git hygiene pass. The untracked fish project should be explicitly included or excluded before the first push so the public repo matches the owner's intent.
