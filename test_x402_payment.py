"""
Test x402 paid endpoint — makes a real $0.001 USDC payment on Base.

Usage:
    python3 test_x402_payment.py <PRIVATE_KEY>

Requirements:
    - Wallet must have >= 0.001 USDC on Base (chain ID 8453)
    - pip install x402 eth-account httpx
"""

import os
import sys
import json
import base64
import httpx
from eth_account import Account
from x402 import x402ClientSync, parse_payment_required
from x402.mechanisms.evm.exact import ExactEvmClientScheme
from x402.mechanisms.evm.signers import EthAccountSigner

API_URL = "https://web3-signals-api-production.up.railway.app/signal"
_INTERNAL_KEY = os.getenv("INTERNAL_API_KEY", "")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 test_x402_payment.py <PRIVATE_KEY>")
        print("\nWallet needs >= 0.001 USDC on Base (chain 8453)")
        sys.exit(1)

    private_key = sys.argv[1]

    # 1. Create signer from private key
    account = Account.from_key(private_key)
    signer = EthAccountSigner(account)
    print(f"Wallet: {account.address}")

    # 2. Create x402 client with EVM exact payment scheme
    x402_client = x402ClientSync()
    x402_client.register("eip155:8453", ExactEvmClientScheme(signer))

    print(f"\nStep 1: Calling {API_URL} ...")

    _headers = {}
    if _INTERNAL_KEY:
        _headers["x-internal-key"] = _INTERNAL_KEY

    with httpx.Client(timeout=60, headers=_headers) as http:
        # 3. Make initial request — expect 402
        resp = http.get(API_URL)
        print(f"  Response: {resp.status_code}")

        if resp.status_code != 402:
            print(f"  Expected 402, got {resp.status_code}")
            if resp.status_code == 200:
                print("  x402 payment gate not active — got data for free")
            print(resp.text[:300])
            return

        # 4. Parse the payment-required header
        payment_header = resp.headers.get("payment-required", "")
        print(f"  Payment header length: {len(payment_header)} chars")

        payment_required = parse_payment_required(
            base64.b64decode(payment_header)
        )
        print(f"  x402 version: {payment_required.x402_version}")
        print(f"  Amount: {payment_required.accepts[0].amount} (raw)")
        print(f"  Network: {payment_required.accepts[0].network}")
        print(f"  Pay to: {payment_required.accepts[0].pay_to}")

        # 5. Create signed payment payload
        print(f"\nStep 2: Signing payment...")
        payment_payload = x402_client.create_payment_payload(payment_required)
        print(f"  Payment payload created (scheme: {payment_payload.accepted.scheme})")

        # 6. Encode payload and retry with payment header
        payload_json = payment_payload.model_dump_json()
        payload_b64 = base64.b64encode(payload_json.encode()).decode()

        print(f"\nStep 3: Retrying with payment...")
        resp2 = http.get(
            API_URL,
            headers={"x-payment": payload_b64},
        )
        print(f"  Response: {resp2.status_code}")

        if resp2.status_code == 200:
            data = resp2.json()
            print(f"\n{'='*60}")
            print(f"  PAYMENT SUCCESSFUL!")
            print(f"{'='*60}")
            print(f"  Timestamp: {data.get('timestamp', 'N/A')}")
            portfolio = data.get("data", {}).get("portfolio_summary", {})
            print(f"  Regime: {portfolio.get('market_regime', 'N/A')}")
            print(f"  Risk: {portfolio.get('risk_level', 'N/A')}")
            signals = data.get("data", {}).get("signals", {})
            print(f"  Assets: {len(signals)}")
            for asset, sig in list(signals.items())[:5]:
                score = sig.get("composite_score", "?")
                direction = sig.get("direction", "?")
                print(f"    {asset}: {score}/100 — {direction}")
            if len(signals) > 5:
                print(f"    ... and {len(signals) - 5} more")
            print(f"\n  First x402 payment complete!")
            print(f"  Bazaar + x402list.fun should auto-list shortly.")
        else:
            print(f"  Payment failed: {resp2.status_code}")
            print(f"  {resp2.text[:500]}")


if __name__ == "__main__":
    main()
