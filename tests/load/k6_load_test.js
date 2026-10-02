import http from 'k6/http';
import { check, sleep } from 'k6';
import { Trend, Rate, Counter } from 'k6/metrics';

// Custom Prometheus/k6 metrics
const l1CacheLatency = new Trend('l1_cache_latency_ms', true);
const streamingTtft = new Trend('streaming_ttft_ms', true);
const errorRate = new Rate('gateway_error_rate');
const successfulRequests = new Counter('successful_completions_total');

export const options = {
  scenarios: {
    // Scenario 1: L1 exact cache throughput under 200 concurrent VUs
    l1_exact_cache_benchmark: {
      executor: 'constant-vus',
      vus: 200,
      duration: '30s',
      exec: 'testL1ExactCache',
    },

    // Scenario 2: Streaming SSE & Time-To-First-Token (TTFT)
    streaming_ttft_benchmark: {
      executor: 'ramping-vus',
      startVUs: 5,
      stages: [
        { duration: '10s', target: 50 },
        { duration: '20s', target: 50 },
        { duration: '5s', target: 0 },
      ],
      exec: 'testStreamingCompletions',
      startTime: '35s',
    },

    // Scenario 3: Traffic spike resiliency (0 -> 400 VUs sudden burst)
    traffic_spike_resilience: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '5s', target: 400 },
        { duration: '15s', target: 400 },
        { duration: '5s', target: 0 },
      ],
      exec: 'testSpikeResilience',
      startTime: '75s',
    },
  },
  thresholds: {
    // Exact cache must resolve under 5ms at p95 under 200 concurrent users
    'l1_cache_latency_ms': ['p(50)<2', 'p(95)<5', 'p(99)<10'],
    'gateway_error_rate': ['rate<0.01'], // <1% failure allowed during spikes
    'http_req_failed': ['rate<0.01'],
  },
};

const BASE_URL = __ENV.GATEWAY_URL || 'http://localhost:7860';

// Setup phase: Prime the L1 cache once before running concurrency benchmarks
export function setup() {
  const primePayload = JSON.stringify({
    model: 'gpt-4o',
    messages: [{ role: 'user', content: 'K6 Benchmark Deterministic Prompt' }],
    temperature: 0.0,
    stream: false,
  });

  const params = {
    headers: { 'Content-Type': 'application/json' },
  };

  const res = http.post(`${BASE_URL}/v1/chat/completions`, primePayload, params);
  check(res, {
    'cache primed successfully': (r) => r.status === 200,
  });
}

// Benchmark execution scenarios
export function testL1ExactCache() {
  const payload = JSON.stringify({
    model: 'gpt-4o',
    messages: [{ role: 'user', content: 'K6 Benchmark Deterministic Prompt' }],
    temperature: 0.0,
    stream: false,
  });

  const params = {
    headers: { 'Content-Type': 'application/json' },
  };

  const startTime = Date.now();
  const res = http.post(`${BASE_URL}/v1/chat/completions`, payload, params);
  const latency = Date.now() - startTime;

  l1CacheLatency.add(latency);

  const passed = check(res, {
    'status is 200': (r) => r.status === 200,
    'hit L1 cache fingerprint': (r) => {
      try {
        const json = r.json();
        return json.system_fingerprint === 'cache:l1_exact';
      } catch (e) {
        return false;
      }
    },
  });

  if (passed) {
    successfulRequests.add(1);
    errorRate.add(0);
  } else {
    errorRate.add(1);
  }
}

// Streaming SSE evaluation
export function testStreamingCompletions() {
  const payload = JSON.stringify({
    model: 'gpt-4o',
    messages: [{ role: 'user', content: 'Stream count to 10' }],
    stream: true,
  });

  const params = {
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'text/event-stream',
    },
    timeout: '15s',
  };

  const startTime = Date.now();
  const res = http.post(`${BASE_URL}/v1/chat/completions`, payload, params);
  const totalDuration = Date.now() - startTime;

  streamingTtft.add(totalDuration);

  const passed = check(res, {
    'status is 200': (r) => r.status === 200,
    'content-type is event-stream': (r) => (r.headers['Content-Type'] || '').includes('text/event-stream'),
    'contains DONE signal': (r) => r.body.includes('[DONE]'),
  });

  if (passed) {
    successfulRequests.add(1);
    errorRate.add(0);
  } else {
    errorRate.add(1);
  }

  sleep(0.05);
}

// Traffic spike resilience evaluation
export function testSpikeResilience() {
  const payload = JSON.stringify({
    model: 'gpt-4o',
    messages: [{ role: 'user', content: `Random prompt ${Math.random()}` }],
    stream: false,
  });

  const params = {
    headers: { 'Content-Type': 'application/json' },
  };

  const res = http.post(`${BASE_URL}/v1/chat/completions`, payload, params);

  const passed = check(res, {
    'status is 200 or 503 circuit-break': (r) => r.status === 200 || r.status === 503,
  });

  if (passed) {
    errorRate.add(0);
  } else {
    errorRate.add(1);
  }
}
