<div align="center">

<img src="assets/readme/hero.svg" width="100%" alt="MARKET VISION / MT5 technical cover" />

**RGB representation of price and indicator windows for image-based research**

[Architecture](#architecture) · [Repository map](#repository-map) · [Run](#getting-started) · [Validation](#validation-and-evidence) · [Roadmap](#engineering-roadmap)

</div>

> **Evidence status: EXPERIMENTAL ML.** This README describes repository contents and boundaries, not verified production readiness or investment outcomes. Previous narrative is preserved in [legacy documentation](docs/README_LEGACY.md) for traceability, not as independent proof.

## Overview

RGB representation of price and indicator windows for image-based research. The purpose is to provide a research and engineering basis that can be inspected and improved without implying performance that has not been independently validated.

| Field | Details |
|:--|:--|
| Repository | [`computer_vision_trading_metatrader_5`](https://github.com/Shtenco/computer_vision_trading_metatrader_5) |
| Primary stack | Python, MetaTrader5, computer vision, neural modeling |
| Role | EXPERIMENTAL ML |
| Source of truth | Code on `main`; experiments require separate reproducible artifacts |
| Parent ecosystem | [SYNERGY SYSTEM](https://github.com/Shtenco/synergy_system) |

## Architecture

```mermaid
flowchart LR
 A[Inputs and configuration] --> B[Repository source]
 B --> C[Research or presentation outputs]
 C --> D[Independent verification gate]
 D --> E[Published evidence, if validated]
```

**Implementation scope:** RGB representation of price and indicator windows for image-based research. The diagram is a conceptual map and does not assert that the verification gate is implemented or automated.

## Repository map

| Path | Purpose |
|:--|:--|
| `computer_vision_trading.py` | Primary executable and research logic |
| `docs/README_LEGACY.md` | Preserved historical documentation; claims may be unverified |
| `assets/readme/hero.svg` | Vector cover, with no external image-host dependencies |

## Getting started

Check required Python modules, vendor platform services and API keys before execution. **Commands below identify the source entry point; they are not evidence of a successful run in this session.**

```bash
git clone https://github.com/Shtenco/computer_vision_trading_metatrader_5.git
cd computer_vision_trading_metatrader_5
python computer_vision_trading.py
```

Use a configured local environment and a demo/data-only setup where applicable. Do not connect research prototypes to funded trading accounts without separate safety review.

## Validation and evidence

**Known limitation:** No independently verified out-of-sample experiment artifact; model claims require leakage and reproducibility audits.

| Layer | Evidence classification |
|:--|:--|
| Source files | Present in repository tree |
| Automated tests and CI | Not established by this documentation pass |
| Out-of-sample or independent evaluation | Not established by this documentation pass |
| Production readiness | Not established |
| Economic or predictive claims | Must be supported by dated, reproducible evidence |

The terms *prototype*, *simulation*, *backtest*, *live deployment* and *audited result* should never be conflated. Any quoted return, accuracy, latency or capability must be tied to a data provenance record, experiment conditions and a commit.

## Engineering roadmap

Purged walk-forward, image timestamp audit, train-only normalization, tabular baselines.

Recommended completion criteria: testable experiment configuration, pinned runtime dependencies, repeatable CI or equivalent checks, archived output manifests, and a documented failure/limitations section.

## Maintainer and attribution

Maintained in the [Shtenco GitHub portfolio](https://github.com/Shtenco). See `LICENSE` for applicable terms. Read the original repository source and [legacy README](docs/README_LEGACY.md) before treating earlier examples or statistics as verified evidence.
