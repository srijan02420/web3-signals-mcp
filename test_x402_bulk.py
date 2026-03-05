"""
Bulk x402 payment test — fire multiple paid calls to build transaction volume.

Usage:
    python3 test_x402_bulk.py <PRIVATE_KEY> <COUNT>

Each call costs $0.001 USDC on Base. 100 calls = $0.10 total.
"""

import os
import sys
import time
import base64
import httpx
from eth_account import Account
from x402 import x402ClientSync, parse_payment_required
from x402.mechanisms.evm.exact import ExactEvmClientScheme
from x402.mechanisms.evm.signers import EthAccountSigner

_INTERNAL_KEY = os.getenv("INTERNAL_API_KEY", "")

ENDPOINTS = [
    "https://web3-signals-api-production.up.railway.app/signal",
    "https://web3-signals-api-production.up.railway.app/signal/BTC",
    "https://web3-signals-api-production.up.railway.app/signal/ETH",
    "https://web3-signals-api-production.up.railway.app/signal/SOL",
    "https://web3-signals-api-production.up.railway.app/performance/reputation",
]


def make_paid_call(http, x402_client, url):
    """Make a single x402 paid call. Returns (success, status_code, elapsed_ms)."""
    start = time.time()
    try:
        # Step 1: Get 402
        resp = http.get(url)
        if resp.status_code != 402:
            return False, resp.status_code, 0

        # Step 2: Parse and sign
        payment_header = resp.headers.get("payment-required", "")
        payment_required = parse_payment_required(base64.b64decode(payment_header))
        payment_payload = x402_client.create_payment_payload(payment_required)
        payload_b64 = base64.b64encode(payment_payload.model_dump_json().encode()).decode()

        # Step 3: Retry with payment
        resp2 = http.get(url, headers={"payment-signature": payload_b64})
        elapsed = int((time.time() - start) * 1000)
        return resp2.status_code == 200, resp2.status_code, elapsed

    except Exception as e:
        elapsed = int((time.time() - start) * 1000)
        return False, str(e)[:50], elapsed


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 test_x402_bulk.py <PRIVATE_KEY> [COUNT]")
        print("  Each call = $0.001 USDC. Default COUNT = 100 ($0.10 total)")
        sys.exit(1)

    private_key = sys.argv[1]
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    account = Account.from_key(private_key)
    signer = EthAccountSigner(account)
    x402_client = x402ClientSync()
    x402_client.register("eip155:8453", ExactEvmClientScheme(signer))

    cost = count * 0.001
    print(f"Wallet:  {account.address}")
    print(f"Calls:   {count}")
    print(f"Cost:    ${cost:.3f} USDC")
    print(f"Endpoints: {len(ENDPOINTS)} (rotating)")
    print(f"{'='*60}")

    success = 0
    fail = 0
    total_ms = 0

    _headers = {}
    if _INTERNAL_KEY:
        _headers["x-internal-key"] = _INTERNAL_KEY

    with httpx.Client(timeout=120, headers=_headers) as http:
        for i in range(count):
            url = ENDPOINTS[i % len(ENDPOINTS)]
            endpoint = url.split(".app")[1]
            ok, status, ms = make_paid_call(http, x402_client, url)

            if ok:
                success += 1
                total_ms += ms
                marker = "OK"
            else:
                fail += 1
                marker = f"FAIL({status})"

            # Progress line
            pct = (i + 1) / count * 100
            avg_ms = total_ms // max(success, 1)
            print(
                f"  [{i+1:3d}/{count}] {pct:5.1f}%  {marker:10s}  "
                f"{ms:5d}ms  {endpoint:30s}  "
                f"ok={success} fail={fail} avg={avg_ms}ms"
            )

            # Small delay to avoid hammering
            if ok:
                time.sleep(0.5)
            else:
                time.sleep(2)  # longer delay on failure

    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"  Success: {success}/{count}")
    print(f"  Failed:  {fail}/{count}")
    print(f"  Cost:    ${success * 0.001:.3f} USDC")
    if success > 0:
        print(f"  Avg latency: {total_ms // success}ms")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
