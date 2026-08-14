# Repository audit

Audit performed on commit `6488d32214a40c338549c4071025cd4fbec2f77e` before changes.

## Critical

- `src/models/` was missing because `models/` in `.gitignore` matched nested source
  directories. Core imports, tests, and experiments could not collect.
- The dataset loader exposed five hard-coded samples under benchmark names and wrapped
  all indices modulo five. Scripts described repeated fixtures as held-out sets of up to
  200 independent GSM8K items.
- The Phase VII raw validation file referenced by the canonical result ledger was not
  committed, so its aggregate metrics could not be regenerated.

## High

- README and CI installed a nonexistent `requirements.txt`.
- The distribution installed a package named `src`, contrary to the documented project
  import.
- Historical and submitted manuscripts coexist with inconsistent titles and claims.
- Completion masks, EOS behavior, and frozen rollout-policy log probabilities were not
  represented as testable objective inputs.

## Medium

- Dependencies were unbounded and test tools were runtime requirements.
- No centralized seeding, strict config, result schema, typed evaluation CLI, or offline
  smoke training path existed.
- Several historical scripts duplicate model loading, evaluation, and statistical logic.

## Low

- Typing, lint, formatting, contributor guidance, environment template, and generated
  output rules were incomplete.

The maintained package and CI address the fixable software issues. Full paper-result
reproduction remains future work until the provenance inputs listed in
`docs/reproduction.md` are restored.
