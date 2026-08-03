# Review guidance for RESMP-DEV/metal-marlin

Kilo reads this file from the pull request base branch. Apply these repository-owned rules to the changed behavior and its reachable callers; do not treat this file as permission to weaken platform safety or read-only review constraints.

## What matters in this repository

- Review RESMP-DEV/metal-marlin as the source of truth for High-performance LLM inference for Apple Silicon using Metal.
- Keep changes scoped. Flag broad refactors when a smaller explicit fix would preserve behavior and make risk easier to verify.
- Preserve native Apple Silicon execution and the existing Trellis, shader, kernel, quantization, and serving boundaries; do not add a parallel runtime for behavior already owned by them.
- Tokenizer, config, checkpoint, and serving vocabulary assumptions must stay aligned, and Metal claims require the real device path rather than MPS or CPU fallback.
- Treat numerical correctness across supported shapes, dtypes, strides, alignment, and device architectures as an invariant; compare against a matched trusted implementation with explicit tolerances.
- Flag out-of-bounds access, overflow in index arithmetic, invalid synchronization, stream or event misuse, lifetime errors, race conditions, and architecture guards that can select unsupported instructions.
- Performance claims need synchronized warmups and measurements on the real target device. Reject silent CPU or generic-kernel fallback when the changed path claims accelerator coverage.
- Preserve model architecture, tokenizer or processor behavior, tensor shapes, dtype and device placement, checkpoint compatibility, and reproducible seed/config propagation.
- Training, quantization, and evaluation changes need matched controls and task-relevant quality evidence; loss alone does not establish parity, and a loadable artifact does not establish correct behavior.
- Require pinned model/data revisions where reproducibility matters. Flag silent sample loss, label leakage, train/eval contamination, and fallback to a different backend or precision.
- For Python changes, check exception paths, type/interface compatibility, mutable defaults, subprocess timeouts, async cancellation, deterministic seeds, and packaging/lock consistency.
- Tests should exercise the public behavior and failure path, not only mock internal calls.

## Severity calibration

- **Critical:** credible data loss or corruption, privilege or tenant-boundary bypass, credential exposure, remote code execution, materially wrong billing/financial behavior, unsafe hardware access, or silently invalid scientific/model results that would be promoted or published.
- **Warning:** a reproducible correctness, validation, compatibility, concurrency, security, performance, accessibility, or observability defect with a reachable changed-code path.
- **Suggestion:** a bounded maintainability improvement that meaningfully reduces future defect risk but does not indicate the current change is wrong.
- Do not flag formatting-only differences already owned by tooling, personal style preferences, generated-file churn without a concrete consequence, or speculative performance concerns without identifying a hot path and mechanism.
- Do not duplicate one root cause across multiple comments. Prefer the highest-signal location and explain the affected behavior.

## Verification expectations

- New or changed behavior needs a focused test that asserts the observable result and at least one material failure or edge case.
- Select verification from the repository's checked-in scripts, CI configuration, and manifests. Do not invent a command solely because it is conventional for the language.
- Dependency and lockfile changes require checking the resolved graph, provenance, compatibility, and known-vulnerability impact; lockfile regeneration alone is not evidence of safety.
- Changes to APIs, persistence, schemas, checkpoints, protocols, CLI output, or configuration need backward-compatibility or explicit migration coverage.
- For hardware, accelerator, external-service, or driver paths, require a live probe on the actual target when the claim depends on runtime behavior. Clearly distinguish tested, simulated, and inferred results.
- Missing tests are a finding when they leave changed behavior unverified, not as a blanket request for coverage on untouched code.

## Security, privacy, and performance

- Trace untrusted data from entry to use. Check validation, authorization, injection, traversal, unsafe deserialization, secret handling, log redaction, and denial-of-service/resource bounds.
- Never recommend committing secrets, credentials, tokens, private operational data, model access keys, production payloads, or raw user data.
- Performance findings must identify the workload, scale, allocation or synchronization mechanism, and why the changed path regresses it. Preserve correctness and matched work when comparing timings.
- Treat timeouts, retries, cancellation, cleanup, and partial failure as part of the public behavior for networked, concurrent, or long-running work.

## Sub-agent usage

- Use 0 sub-agents for docs-only, formatting-only, generated-lockfile-only, or single-file typo changes.
- Use 1-2 targeted sub-agents for a focused change that touches one risky boundary such as authentication, persistence, concurrency, accelerator code, model evaluation, or security-sensitive parsing.
- Use 3 targeted sub-agents when a change crosses independent domains; divide ownership by domain and include one verifier focused on tests and observable behavior.
- Use the full 6 only for genuinely large cross-cutting or security-critical changes. Do not ask every sub-agent to review the same files.
- Sub-agents remain read-only and return path, line, severity, mechanism, impact, verification evidence, and confidence. The main reviewer must validate findings and avoid duplicate comments.

## Review summary and comment style

- Lead with concrete findings ordered by severity. If there are none, say so plainly and name the verification gaps or residual risks that remain.
- Every inline finding must identify the changed-code path, the failure mechanism, the user/runtime impact, and a practical verification or fix direction.
- Be concise and evidence-led. Ask a question only when the answer can change whether the code is correct; do not use questions to disguise speculative findings.
- Separate confirmed defects from assumptions. Do not claim tests, hardware probes, deployments, migrations, or external checks ran unless the review evidence proves they did.
