# Third-Party Notices

This document inventories the third-party software used by slime-das. It must
be updated whenever `requirements.txt` or vendored source code changes.

## Upstream project

| Component | Source | Version / commit | License | Local use |
| --- | --- | --- | --- | --- |
| slime | https://github.com/THUDM/slime | `v0.3.0` / `bf14dc21f9500746447f2572d0692e981c4d2a7e` | Apache-2.0 | Upstream baseline for this repository |

slime-das retains the applicable upstream copyright notices and license terms.
HYGON modifications are identified in the relevant source files.

## Python dependencies

The following are direct dependencies declared in `requirements.txt`. Entries
marked **Pending** must be completed with an approved, fixed version and the
license verified from the corresponding released distribution before a formal
open-source release.

| Component | Source | Version | License | Local use / status |
| --- | --- | --- | --- | --- |
| accelerate | https://pypi.org/project/accelerate/ | Pending lock | Pending verification | Python dependency |
| blake3 | https://pypi.org/project/blake3/ | Pending lock | Pending verification | Python dependency |
| blobfile | https://pypi.org/project/blobfile/ | Pending lock | Pending verification | Python dependency |
| codetiming | https://pypi.org/project/codetiming/ | Pending lock | Pending verification | Python dependency |
| datasets | https://pypi.org/project/datasets/ | Pending lock | Pending verification | Python dependency |
| gguf | https://pypi.org/project/gguf/ | Pending lock | Pending verification | Python dependency |
| httpx | https://pypi.org/project/httpx/ | Pending lock | Pending verification | Python dependency (`http2` extra) |
| hydra-core | https://pypi.org/project/hydra-core/ | Pending lock | Pending verification | Python dependency |
| mathruler | https://pypi.org/project/mathruler/ | Pending lock | Pending verification | Python dependency |
| mistral_common | https://pypi.org/project/mistral-common/ | Pending lock | Pending verification | Python dependency |
| msgspec | https://pypi.org/project/msgspec/ | Pending lock | Pending verification | Python dependency |
| numba | https://pypi.org/project/numba/ | Pending lock | Pending verification | Python dependency |
| omegaconf | https://pypi.org/project/omegaconf/ | Pending lock | Pending verification | Python dependency |
| openai | https://pypi.org/project/openai/ | Pending lock | Pending verification | Python dependency |
| partial_json_parser | https://pypi.org/project/partial-json-parser/ | Pending lock | Pending verification | Python dependency |
| peft | https://pypi.org/project/peft/ | Pending lock | Pending verification | Python dependency |
| py_cpuinfo | https://pypi.org/project/py-cpuinfo/ | Pending lock | Pending verification | Python dependency |
| pybase64 | https://pypi.org/project/pybase64/ | Pending lock | Pending verification | Python dependency |
| pylatexenc | https://pypi.org/project/pylatexenc/ | Pending lock | Pending verification | Python dependency |
| pyyaml | https://pypi.org/project/PyYAML/ | Pending lock | Pending verification | Python dependency |
| qwen_vl_utils | https://pypi.org/project/qwen-vl-utils/ | Pending lock | Pending verification | Python dependency |
| ray | https://pypi.org/project/ray/ | `>=2.56.0` | Pending verification | Python dependency (`default` extra) |
| safetensors | https://pypi.org/project/safetensors/ | Pending lock | Pending verification | Python dependency |
| sentencepiece | https://pypi.org/project/sentencepiece/ | Pending lock | Pending verification | Python dependency |
| sglang-router | https://pypi.org/project/sglang-router/ | `0.3.2` | Pending verification | Python dependency |
| tensorboard | https://pypi.org/project/tensorboard/ | Pending lock | Pending verification | Python dependency |
| tensordict | https://pypi.org/project/tensordict/ | Pending lock | Pending verification | Python dependency |
| tiktoken | https://pypi.org/project/tiktoken/ | Pending lock | Pending verification | Python dependency |
| torch_memory_saver | https://pypi.org/project/torch-memory-saver/ | Pending lock | Pending verification | Python dependency |
| torchdata | https://pypi.org/project/torchdata/ | Pending lock | Pending verification | Python dependency |
| transformers | https://pypi.org/project/transformers/ | Pending lock | Pending verification | Python dependency |
| wandb | https://pypi.org/project/wandb/ | Pending lock | Pending verification | Python dependency |
| xgrammar | https://pypi.org/project/xgrammar/ | Pending lock | Pending verification | Python dependency |

## Git dependency

| Component | Source | Version / commit | License | Local use / status |
| --- | --- | --- | --- | --- |
| mbridge | https://pypi.org/project/mbridge/ | `0.15.1` | Apache-2.0 | Installed by `requirements.txt`; published from `ISEEKYAN/mbridge@0cd4ae23f2425da77a80cb3f517828452fa8e984`; no source is vendored in this repository |

## Release checklist

Before release, for every dependency listed above:

1. Replace `Pending lock` with the exact approved version in both this file and
   `requirements.txt`.
2. Replace `Pending verification` with the SPDX license identifier verified
   from the exact released distribution or source commit.
3. Record any copied or vendored third-party source separately with its exact
   local path and modification status.
