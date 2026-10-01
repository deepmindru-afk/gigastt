# Homebrew release updates

After the Release workflow succeeds on a version tag, Homebrew automation reads the published
`SHA256SUMS.txt` and proposes only `Formula/gigastt.rb` in a release-specific PR.
Manual runs accept an existing release tag. Missing, duplicate or malformed
checksums stop the update before the formula is written. Repeated runs reuse the
same branch and PR. The formula keeps its existing pins until that PR is merged.
A manual selection of an older release can propose a downgrade; review the
version as well as the checksums.

The repository must enable **Settings → Actions → General → Workflow permissions
→ Allow GitHub Actions to create and approve pull requests**. The workflow requests
`contents: write` and `pull-requests: write`; it never approves reviews, merges,
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
