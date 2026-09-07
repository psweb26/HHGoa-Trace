"""Ethereum-compatible local-chain storage and verification."""

from __future__ import annotations

from dataclasses import dataclass

from eth_tester import EthereumTester, PyEVMBackend
from web3 import Web3
from web3.providers.eth_tester import EthereumTesterProvider


class BlockchainError(RuntimeError):
    """Raised when the local Ethereum transaction cannot be stored or read."""


@dataclass(frozen=True)
class ChainReceipt:
    transaction_hash: str
    block_number: int
    stored_fingerprint: str


class LocalEthereumVerifier:
    """Stores one 32-byte SHA-256 digest in a mined eth-tester transaction's data."""

    def __init__(self) -> None:
        try:
            self.tester = EthereumTester(backend=PyEVMBackend())
            self.web3 = Web3(EthereumTesterProvider(self.tester))
        except Exception as exc:  # Dependency errors should be readable to CLI users.
            raise BlockchainError(
                "Could not initialize eth-tester/PyEVM. Install requirements.txt, "
                "including the eth-tester[py-evm] extra."
            ) from exc

    def write_fingerprint(self, fingerprint: str) -> ChainReceipt:
        _validate_fingerprint(fingerprint)
        try:
            sender = self.web3.eth.accounts[0]
            tx_hash = self.web3.eth.send_transaction(
                {
                    "from": sender,
                    "to": sender,
                    "value": 0,
                    "data": "0x" + fingerprint,
                    "gas": 100_000,
                }
            )
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
        except Exception as exc:
            raise BlockchainError(f"Could not write fingerprint to local Ethereum chain: {exc}") from exc

        return ChainReceipt(
            transaction_hash=tx_hash.to_0x_hex(),
            block_number=int(receipt["blockNumber"]),
            stored_fingerprint=fingerprint.lower(),
        )

    def read_fingerprint(self, transaction_hash: str) -> str:
        try:
            transaction = self.web3.eth.get_transaction(transaction_hash)
            transaction_input = transaction.get("input", transaction.get("data", ""))
        except Exception as exc:
            raise BlockchainError(f"Could not read transaction '{transaction_hash}' from local chain: {exc}") from exc

        if isinstance(transaction_input, bytes):
            encoded = transaction_input.hex()
        else:
            encoded = str(transaction_input)
            if encoded.startswith("0x"):
                encoded = encoded[2:]
        _validate_fingerprint(encoded)
        return encoded.lower()

    def verify(self, transaction_hash: str, local_fingerprint: str) -> tuple[bool, str]:
        _validate_fingerprint(local_fingerprint)
        on_chain = self.read_fingerprint(transaction_hash)
        return on_chain == local_fingerprint.lower(), on_chain


def _validate_fingerprint(fingerprint: str) -> None:
    if len(fingerprint) != 64 or any(character not in "0123456789abcdefABCDEF" for character in fingerprint):
        raise BlockchainError("Fingerprint must be a 64-character SHA-256 hexadecimal digest.")
