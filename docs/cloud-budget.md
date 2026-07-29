# Sub-$100 cloud runbook

The dollar ceiling is a hard stop, not a target. The default allocation totals
$100, while profile selection permits at most $90 of projected compute so the
run retains a small uncertainty margin. Provider price and availability change;
enter the displayed launch-time rate rather than copying an old advertised
price.

> **Scaffold boundary:** the checked-in cloud script is a dry-run preflight.
> It validates cost inputs and prints the resolved plan, but it does not rent a
> GPU or start paid training. Implement and validate the measured probe,
> prepared-data handoff, mid-stage resume, and watchdog integration before
> using this runbook to authorize spend.

## Allocation

| Stage | Maximum |
| --- | ---: |
| Data/tokenizer and throughput probe | $5 |
| Base training | $62 |
| SFT | $8 |
| GRPO | $17 |
| Evaluation/retry reserve | $6 |
| Short-lived storage | $2 |
| **Total** | **$100** |

Taxes, egress, minimum billing increments, and detached storage may sit outside
a provider's GPU hourly price. Check them before launch. If they cannot fit
inside the total, lower the config cap; do not assume they are free.

## Before implementing a paid run

1. Run `./scripts/run_local_smoke.sh` from a clean environment.
2. Confirm exact checkpoint resume tests pass.
3. Prepare or identify persistent storage and test that it survives compute
   termination. Set `NANOGPT_HOME` to that mount.
4. Copy `.env.example` to an untracked `.env` or export variables directly.
5. Run the cloud script and inspect every resolved path,
   dataset revision, work target, and budget.
6. Check the provider's exact H100 type, memory, region, storage, egress, and
   hourly price. Export that rate as `NANOGPT_HOURLY_RATE_USD`.

Never bake `HF_TOKEN` or provider credentials into an image, config, shell
history, log, or checkpoint manifest.

## Measured profile selection

On the target device, warm up compilation and run the 10M-token probe from
`configs/run/cloud_probe.yaml`. Let measured throughput be `q` tokens/second,
remaining target work `N`, hourly rate `r`, and safety margin `m=0.20`:

```text
projected_hours = (N / q) / 3600 * (1 + m)
projected_cost  = projected_hours * r
```

SFT and evaluation use their measured token throughput. GRPO uses measured
prompt-rollout throughput because generation, not optimizer tokens, dominates.
Sum stage projections plus incurred cost and storage. Choose the 100M/2B-token
profile only when projected compute is at most $90 and every stage cap fits.
Otherwise choose 50M/1B. If neither fits, reduce the token horizon and record a
budget-limited checkpoint; never lower the safety margin just to pass the gate.

## Cost watchdog

Every process receives the exact hourly rate, a stage cap, the already-incurred
project cost, and total cap. It samples monotonic elapsed time at least every 15
seconds:

```text
estimated_stage_cost = elapsed_seconds / 3600 * hourly_rate
remaining_seconds = (stage_cap - estimated_stage_cost) / hourly_rate * 3600
```

When remaining time reaches the configured 120-second margin, stop accepting
new work, atomically checkpoint, flush metrics, and exit with a recognizable
`budget_exhausted` status. The manifest records actual start/end timestamps,
monotonic duration, price input, estimated cost, and cap. It must not claim
provider-invoice precision.

The orchestration script stops after any failed or budget-exhausted stage. It
does not retry, start the next paid stage, or destroy storage automatically.

## Preflight and future launch

The implemented script performs inspection only:

```bash
NANOGPT_HOURLY_RATE_USD=2.99 ./scripts/run_cloud_100.sh
```

`--execute` is reserved and currently exits before any paid action. The next
implementation phase must connect the prepared artifacts, measured profile
selection, stage watchdog, and resume loader, then add tests that simulate
preemption and budget exhaustion before enabling this command:

```bash
NANOGPT_HOME=/persistent/nanogpt \
NANOGPT_HOURLY_RATE_USD=2.99 \
NANOGPT_TOTAL_BUDGET_USD=100 \
./scripts/run_cloud_100.sh --execute
```

Once implemented, a preempted run should attach a new instance to the same
persistent volume, verify checkpoint hashes, inspect `run_manifest.json`, and
invoke the explicit resume mode. The loader must reject changed
config/data/tokenizer/parent hashes. Do not use spot capacity until this path
has been proven with simulated interruption and the tiny pipeline.

## Shutdown checklist

- Confirm the final atomic checkpoint and evaluation report exist on persistent
  storage and verify their hashes from a second process.
- Copy small manifests, configs, metrics, and logs to durable backup.
- Record provider usage and any taxes/egress separately from the code estimate.
- Terminate the GPU instance immediately; verify it is stopped, not merely
  disconnected.
- Delete unattached disks only after backup verification, then confirm no idle
  IP, volume, snapshot, or endpoint remains billable.
- Never publish prepared web-data shards or trusted-local `trainer_state.pt`.
