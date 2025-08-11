import argparse
import asyncio
import time
import json
from statistics import mean, median, quantiles

import httpx

async def worker(task_queue: asyncio.Queue, results: list, method: str, url: str, body):
    async with httpx.AsyncClient() as client:
        while True:
            _ = await task_queue.get()
            if _ is None:
                task_queue.task_done()
                break
            start = time.perf_counter()
            try:
                if method.upper() == "POST":
                    resp = await client.post(url, json=body, timeout=30.0)
                else:
                    resp = await client.get(url, timeout=30.0)
                elapsed = time.perf_counter() - start
                results.append((elapsed, resp.status_code))
            except Exception as e:
                elapsed = time.perf_counter() - start
                results.append((elapsed, None))
            task_queue.task_done()

async def run_test(url: str, concurrency: int, total: int, method: str, body):
    task_queue = asyncio.Queue()
    results = []

    # enqueue tasks
    for _ in range(total):
        await task_queue.put(url)
    for _ in range(concurrency):
        await task_queue.put(None)

    # start workers
    workers = [
        asyncio.create_task(worker(task_queue, results, method, url, body))
        for _ in range(concurrency)
    ]
    start = time.perf_counter()
    await task_queue.join()
    duration = time.perf_counter() - start

    for w in workers:
        w.cancel()

    times = [r[0] for r in results]
    statuses = [r[1] for r in results]
    success = sum(1 for s in statuses if s and 200 <= s < 300)
    failures = total - success

    print(f"Endpoint: {url}")
    print(f"Method: {method.upper()}")
    if method.upper() == "POST":
        print(f"Request body: {body}")
    print(f"Total requests: {total}")
    print(f"Concurrency: {concurrency}")
    print(f"Total time: {duration:.2f}s")
    print(f"Requests/sec: {total/duration:.2f}")
    print(f"Success: {success}, Failures: {failures}")
    print(f"Avg latency: {mean(times)*1000:.1f}ms")
    print(f"Median latency: {median(times)*1000:.1f}ms")
    pcts = quantiles(times, n=100)
    print(f"90th percentile latency: {pcts[89]*1000:.1f}ms")
    print(f"99th percentile latency: {pcts[98]*1000:.1f}ms")

def main():
    parser = argparse.ArgumentParser(description="Async stress test for FastAPI endpoints")
    parser.add_argument("--url", required=True, help="Full URL to hit")
    parser.add_argument("--method", choices=["GET", "POST"], default="POST", help="HTTP method")
    parser.add_argument("--body", help="JSON string for POST body, e.g. '{\"user_id\":\"...\"}'")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrent workers")
    parser.add_argument("--total", type=int, default=100, help="Total requests to send")
    args = parser.parse_args()

    body = None
    if args.method == "POST":
        if not args.body:
            parser.error("POST method requires --body.")
        try:
            body = json.loads(args.body)
        except json.JSONDecodeError:
            parser.error("Invalid JSON for --body.")

    asyncio.run(run_test(args.url, args.concurrency, args.total, args.method, body))

if __name__ == "__main__":
    main()
