# Homebrew release updates

After the Release workflow succeeds, Homebrew automation reads its validated tag
metadata and the published
`SHA256SUMS.txt` and proposes only `Formula/gigastt.rb` in a release-specific PR.
Manual runs accept an existing release tag. Missing, duplicate or malformed
checksums stop the update before the formula is written. Repeated runs reuse the
same branch and PR. The formula keeps its existing pins until that PR is merged.
A manual selection of an older release can propose a downgrade; review the
version as well as the checksums.

The repository must enable **Settings → Actions → General → Workflow permissions
→ Allow GitHub Actions to create and approve pull requests**. The workflow requests
`actions: read` for the metadata handoff and `contents: write` /
`pull-requests: write` only in the proposal job; it never approves reviews, merges,
pushes directly to `main`, or changes branch protection. No additional secret is
required.

GitHub places checks for PRs created or updated with `GITHUB_TOKEN` in an
approval-required state. A maintainer with write access selects **Approve
workflows to run** on the PR, reviews the generated pins and waits for every
required check before merging. Do not treat the generator workflow succeeding
as proof that the formula PR passed CI. See the
[GitHub token event rules](https://docs.github.com/en/actions/concepts/security/github_token#when-github_token-triggers-workflow-runs).

The commit-pinned `peter-evans/create-pull-request` action handles repeatable
branch and PR updates, avoiding a separate custom Git/PR state machine. Its
`add-paths` setting restricts commits to the formula; downloaded assets are kept
outside the checkout.

## Manual releases and reruns

A Release dispatched from `main` with an existing version tag uses the same
Homebrew path as a tag push. The dispatch branch is not the release tag.
The release resolver uploads a small JSON artifact containing its validated tag,
immutable source commit, repository, run ID and attempt. Homebrew downloads only
that artifact from the triggering successful same-repository Release run and
requires the exact current attempt. It executes scripts from `main`, never from
the artifact. Per-tag job concurrency and the deterministic PR branch also apply
to explicit manual Homebrew runs.

Missing, expired or invalid metadata stops automatic proposals. Before this
handoff is deployed, or when releasing a tag whose workflow predates it, manually
run **Homebrew Formula update** with the published release tag. That is also the
fallback after **Re-run failed jobs** if GitHub reused the successful resolver
from an earlier attempt: its old metadata is deliberately not accepted for the
new attempt. Alternatively, **Re-run all jobs** produces fresh metadata, but also
repeats the Release workflow's normal build and publication operations. Metadata
is retained for seven days; the manual Homebrew path reads published checksums
and does not depend on that artifact.

The consumer is triggered by `workflow_run` and must already exist on the default
branch. See [GitHub workflow-run semantics](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#workflow_run).
