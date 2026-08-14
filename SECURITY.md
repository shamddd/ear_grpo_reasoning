# Security

The verified smoke path requires no credentials. Store optional provider credentials in
environment variables or an untracked `.env` file; never commit tokens, model caches, or
private dataset URLs. If a credential is discovered in Git history, revoke it before
attempting history cleanup.

Report vulnerabilities privately to the repository owner through GitHub's security
reporting interface when available.
