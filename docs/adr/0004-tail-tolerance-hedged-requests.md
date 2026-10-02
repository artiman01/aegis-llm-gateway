# ADR 0004: Adaptive Speculative Hedged Requests for Long-Tail Latency Mitigation

- **Status:** Accepted
- **Date:** 2026-10-02
- **Author:** Principal Software Architect
- **Deciders:** Core Engineering Team

---

## 1. Context and Problem Statement

In Large Language Model Gateways operating at scale, latency distributions exhibit extreme long tails. While the median Time-To-First-Token (TTFT) for modern commercial models may hover around 250–500ms, the 95th and 99th percentiles often spike to 3000ms–8000ms.

These long-tail latency excursions arise from:
1. **Upstream GPU Queue Stalls:** Variable batch scheduling and preemptive context eviction in multi-tenant inference clusters.
2. **KV-Cache Memory Fragmentation:** Re-allocations on congested cloud nodes.
3. **Transient Network Jitter & Cloud Routing Hops:** Cross-datacenter packet loss and TCP retransmissions.

In high-concurrency environments, a single slow upstream request degrades client throughput and starves connection pools. Traditional retries with exponential backoff exacerbate tail latency, as they wait for full timeout expiration before initiating recovery.

---

## 2. Decision: Adaptive Speculative Hedged Requests (The Tail at Scale)

Drawing from the principles formulated by Jeff Dean and Luiz André Barroso (*"The Tail at Scale"*, CACM 2013), AegisLLM adopts **Adaptive Speculative Hedged Requests** for both non-streaming and streaming completions.

```
Client Request
      │
      ▼
┌─────────────────────────┐
│ Dispatch to Primary     │ ──► Send HTTP/2 Stream to Primary Provider
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Dynamic P90 Timer       │ Wait up to: T_hedge = P90_TTFT(provider) + Margin
└───────────┬─────────────┘
            │
      ┌─────┴─────────────────────────────────────┐
      │ First Token arrived within T_hedge?       │
      ▼ YES                                       ▼ NO (Tail Latency Detected)
┌───────────────────────┐             ┌─────────────────────────┐
│ Stream from Primary   │             │ Speculatively Launch    │
│ to Client             │             │ Fallback Provider       │
└───────────────────────┘             └───────────┬─────────────┘
                                                  │
                                                  ▼
                                      ┌─────────────────────────┐
                                      │ Race to First Token     │
                                      │ (asyncio.wait FIRST)    │
                                      └───────────┬─────────────┘
                                                  │
                                      ┌───────────┴─────────────┐
                                      ▼ Winner                  ▼ Loser
                          ┌───────────────────────┐ ┌───────────────────────┐
                          │ Stream Chunks to      │ │ Cancel Task           │
                          │ Downstream Client     │ │ Send HTTP/2 RST_STREAM│
                          └───────────────────────┘ └───────────────────────┘
```

---

## 3. Dynamic Quantile Estimation Algorithm

To avoid static, misconfigured timeouts, the gateway continuously computes an online rolling $P_{90}$ latency estimate for each provider.

### 3.1 Sliding Window Reservoir
Each registered provider maintains a thread-safe circular reservoir buffer $\mathcal{B}_p$ of capacity $K = 100$ capturing recent TTFT (or completion latency) observations:

$$\mathcal{B}_p = [\tau_1, \tau_2, \dots, \tau_K]$$

When a provider yields its first token in $\tau_{\text{obs}}$ milliseconds, $\tau_{\text{obs}}$ is appended to $\mathcal{B}_p$.

### 3.2 Dynamic Hedging Delay Calculation
The dynamic hedging deadline $T_{\text{hedge}}(p)$ is computed as:

$$P_{90}(p) = \text{Quantile}_{0.90}(\mathcal{B}_p)$$

$$T_{\text{hedge}}(p) = \max\left(T_{\text{min}}, P_{90}(p) + \Delta_{\text{margin}}\right)$$

Where:
- $T_{\text{min}} = 100\,\text{ms}$ prevents premature hedging on fast local or mock providers.
- $\Delta_{\text{margin}} = 50\,\text{ms}$ absorbs expected network jitter before hedging.
- Default baseline prior: If $|\mathcal{B}_p| < 5$, $T_{\text{hedge}}$ defaults to a conservative fallback (e.g. $400\,\text{ms}$).

---

## 4. Streaming Hedging Mechanics & HTTP/2 RST_STREAM Cancellation

Streaming completions present unique challenges because tokens arrive incrementally over Server-Sent Events (SSE).

### 4.1 First-Chunk Racing
1. The gateway dispatches the request to the primary provider and creates an asynchronous task $\mathcal{T}_{\text{primary}}$.
2. Concurrently, an asynchronous timer waits for $T_{\text{hedge}}$.
3. If $\mathcal{T}_{\text{primary}}$ yields its first chunk before $T_{\text{hedge}}$ expires:
   - Primary wins. The timer is cancelled.
   - Chunks are yielded directly to the client.
4. If $T_{\text{hedge}}$ expires without a chunk from Primary:
   - A speculative task $\mathcal{T}_{\text{fallback}}$ is spawned in parallel for the fallback provider.
   - The gateway awaits the first chunk from either $\mathcal{T}_{\text{primary}}$ or $\mathcal{T}_{\text{fallback}}$ using `asyncio.wait(..., return_when=FIRST_COMPLETED)`.

### 4.2 Immediate Loser Cancellation
The moment a winning provider emits a valid first chunk:
1. The winning provider claims stream ownership and passes the "Point of No Return".
2. The losing task is **immediately cancelled**:
   ```python
   loser_task.cancel()
   ```
3. In `httpx.AsyncClient` with HTTP/2 enabled, cancelling the streaming response iterator automatically triggers an **`RST_STREAM` frame** sent to the losing provider server.
4. This terminates upstream token generation, freeing inference resources and avoiding redundant GPU billing.

---

## 5. Consequences and Operational Trade-offs

### Positive Consequences
- **Drastic Reduction in Tail Latency:** Shaves up to 70–80% off the P99 latency experienced by clients during upstream provider degradations.
- **Autonomous Recovery:** Transient node pauses at OpenAI or Anthropic trigger sub-second fallback without user errors or retry storms.
- **Resource Efficiency:** Thanks to HTTP/2 stream multiplexing and immediate `RST_STREAM` cancellation, redundant inference is terminated within tens of milliseconds.

### Negative Consequences / Mitigation
- **Marginal Request Volume Overhead:** Speculative requests dispatched during tail excursions increase total upstream requests by a small percentage (typically $< 3\%$ under 90th percentile thresholding).
  - *Mitigation:* The hedging trigger is bound strictly to $P_{90} + \Delta$, ensuring speculative dispatches only occur when a request is genuinely stalled.
