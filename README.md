# HHGoa-Trace — HH Goa 2026 Task 3

A Python CLI that turns an authorized face-image scan into a verifiable proof:

```text
input image -> MTCNN face detection -> InceptionResnetV1 embedding
-> SerpApi image upload -> Google Lens reverse-image search -> validated public social-media post
-> canonical JSON -> SHA-256 -> local Ethereum-compatible transaction
-> transaction read-back -> SHA-256 recomputation -> VERIFIED / FAILED
```

This is intentionally a command-line project: no website, database, or hidden result set is involved.

## Architecture

![HHGoa-Trace architecture](assets/architecture.png)

```text
Input image
  -> MTCNN face detection
  -> InceptionResnetV1 embedding
  -> SerpApi image upload
  -> Google Lens reverse-image search
  -> validated social-media post
  -> canonical JSON + SHA-256
  -> Ethereum-compatible local transaction (PyEVM)
  -> on-chain read-back + hash comparison
  -> VERIFIED / FAILED
```

## What it does

1. Loads the supplied image, detects all faces with **MTCNN**, and chooses the highest-confidence face (face area breaks ties).
2. Generates a 512-dimensional **InceptionResnetV1 / FaceNet** (`vggface2`) embedding. This is the required face-encoding step; the embedding is not published in the proof artifact.
3. Uploads the actual supplied image to SerpApi's Image API, receives a runtime image ID, and sends that ID to Google Lens. Google Lens performs an `exact_matches` search first and a `visual_matches` search only if no validated social post is found. The embedding is not used as a web-search query.
4. Accepts only a Lens-returned URL that is validated as an individual public social-media post. It requires a platform-specific post URL pattern where known, or conservative content-specific URL plus title/snippet evidence for an unrecognised valid variant. Homepages, profiles, search pages, hashtag pages, login pages, and generic channels are rejected. Exact matches are preferred over visual matches.
5. Writes URL, platform, title, snippet, result section, thumbnail when returned, independently reported similarity when returned, input SHA-256, and non-sensitive face metadata into deterministic canonical JSON.
6. SHA-256 hashes that canonical JSON and stores the 32-byte digest in the `data` field of a mined local Ethereum-compatible transaction.
7. Reads the actual transaction back, recomputes SHA-256 from the saved JSON record, compares the values, and prints `VERIFIED` only on equality.

## Why reverse-image search?

The face model handles detection and encoding. Google Lens performs the actual internet-wide image matching, which is appropriate here because public web/social pages are indexed as images rather than exposed as a universal face-recognition database.

## Setup

Use 64-bit Python 3.11 on Windows (the version used to validate the local-chain dependency path). A virtual environment is recommended:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Create a [SerpApi](https://serpapi.com/) account, copy its API key to `.env`, and replace the placeholder:

```dotenv
SERPAPI_KEY=your_real_key
```

The first face-encoding run may download FaceNet model weights. Use a JPG, PNG, or other Pillow-readable image containing a face that you have permission to process. To make an easy live demonstration, use an image you own that has already been posted publicly; reverse-image search only finds content that the provider can index.

The pinned local-chain packages intentionally install `eth-tester` and its compatible `py-evm` backend directly. Do **not** substitute `eth-tester[py-evm]` on Windows: that optional extra requests `safe-pysha3`, which currently has no CPython 3.11 Windows wheel and can trigger a compiler build. The direct pinned setup uses available wheels and does not require a C++ toolchain.

## Run

```powershell
python main.py .\data\your_public_face_image.jpg
```

The command saves a proof JSON file under `artifacts/` (ignored by Git) and prints actual values, for example:

```text
[1/4] Detecting and encoding face...
      Detected 1 face(s); selected 512-D embedding.
[2/4] Running genuine reverse-image search...
      Live exact_match result selected from Instagram.
[3/4] Writing fingerprint to blockchain...
[4/4] Re-verifying on-chain record...

================ VERIFICATION ================
Search result: https://www.instagram.com/p/returned_by_google_lens/
Platform: Instagram
Match type: exact_match
Result type: social_media_post
Transaction hash: 0xactual_eth_tester_transaction_hash
Block number: 1
Local SHA-256: ...
On-chain SHA-256: ...
VERIFIED: True
================================================
```

The values above illustrate format only; the program never uses them as data. If Lens returns no usable match, face detection fails, or the API key/network is unavailable, it prints the real error and does **not** create a blockchain record.

## Verification methodology

`main.py` saves `{record, canonical_sha256}` to an artifact. It then reopens the saved artifact, serializes `record` with `sort_keys=True` and compact separators, and calculates SHA-256 again. `blockchain.py` retrieves the transaction by its actual transaction hash and compares the transaction input bytes with that recomputed hash. `VERIFIED: True` is emitted only when those values match.

## Blockchain choice

The project uses `eth-tester` with the PyEVM backend: it is a real local Ethereum Virtual Machine implementation, mined transactions, transaction hashes, blocks, and transaction-data read-back—not a custom hash chain. It is suitable for a dependable offline hackathon demo. The chain is process-local, so it is not a public or persistent immutable network; restart the command and a new local chain is created.

## Known limitations and responsible use

- Process images only with permission. This tool reports a reverse-image match; it does not establish a person's identity, ownership, or consent.
- Search coverage depends on SerpApi/Google Lens indexing, account quota, API availability, platform privacy settings, and whether a public page permits crawling. This implementation deliberately fails if it cannot validate an individual public social-media post; it does not fall back to a generic web page or a social profile.
- The SerpApi image-upload API supports JPG/PNG/WebP up to 500 KiB. The program converts and compresses the query image in memory when needed; the canonical record still fingerprints the original supplied file.
- Face embedding quality is affected by pose, lighting, occlusion, detector errors, and model bias. The embedding only validates the face-processing stage and is not itself searched against the web.
- `eth-tester` + `py-evm` is a local, simulated Ethereum-compatible blockchain. This is permitted by the task, but it is process-local rather than a public, persistent network. For durable public proof, replace the provider with a funded public-testnet provider and appropriate account management.

## Repository layout

```text
HHGoa-Trace/
├── main.py
├── face_pipeline.py
├── reverse_search.py
├── blockchain.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── data/
└── artifacts/
```
