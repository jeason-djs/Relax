# Task 22: Sync-intent-aware adaptive oversampling

## Scope

This experiment builds on Relax's existing fully-async over-sampling and partial-rollout implementation
(`377f612`, also evaluated in Issue #152 / PR #153). Those existing capabilities are treated as the common
tail-hedging substrate, not as this change's performance contribution.

The new policy keeps per-step weight publication unchanged and adds:

1. an Actor-to-Rollout synchronization intent;
2. a normal 16-group candidate window that contracts to previous-partition debt while publication is pending;
3. an early TransferQueue close for the debt partition;
4. higher non-preemptive SGLang waiting-queue priority for old-debt requests;
5. automatic restoration of the normal candidate window after publication.

## Difference from related work

- PR #211 reduces weight-publication frequency. This experiment still publishes every step.
- Issue #152 uses fixed partial rollout / over-sampling. This experiment dynamically changes the admission
  window around the publication boundary and gives debt work explicit service priority.
- The earlier Task 22 rolling-transition prototype changes engine weight versions independently. This
  experiment retains the existing global weight update and is implemented in a separate worktree.

## ON contract

```text
rollout_batch_size                 8 groups
normal candidate window           16 groups
partial rollout                   enabled
mask old partial prefix           enabled
maximum repeated aborts           2
weight publication interval       1
old-debt waiting-queue priority   1
fresh/eval priority               0
priority preemption               disabled
```

When no previous-partition debt exists, rollout remains work-conserving. When debt exists and the Actor has
announced a publication intent, only the missing debt groups are admitted. The previous partition is closed
as soon as all debt groups finish, then fresh admission resumes after the weight publication completes.

## Attribution

A matched comparison must use partial rollout and the same 16-group normal candidate window in both arms.
The candidate contribution is only:

- dynamic contraction/restoration of the window;
- debt early close;
- old-debt priority.

Static partial rollout / over-sampling gains must be reported separately as common baseline behavior.
