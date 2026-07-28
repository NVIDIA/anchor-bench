# Anchor Bench SBOM

CycloneDX Software Bill of Materials for `anchor-bench` **0.1.0**.

NVIDIA nSpect expects **CycloneDX 1.6 JSON**
([SBOM Info and User Guide](https://nvidia.atlassian.net/wiki/spaces/PRODSEC/pages/2569287823)).

## Files

| File | Description |
| --- | --- |
| `anchor-bench-0.1.0-python.cdx.json` | **Primary** CycloneDX 1.6 JSON (base deps + installable `full` extras; excludes `dev` tools) |
| `anchor-bench-0.1.0-python.csv` | Spreadsheet view, plus declared deps not installable on this host |
| `requirements-freeze.txt` | `pip freeze` snapshot matching the primary SBOM env |

## Gaps

Not resolved in this host-generated SBOM (listed in the CSV with notes):

- `isaacsim==6.0.0.0`
- `newton==1.2.0`
- `mujoco-warp>=3.5.0`
- Isaac Lab @ `482d21c936b863e323f0a135c8e8e8e8a9c166b2` (`versions/compatibility.toml`)

For the shipping container, prefer an nSpect **container image** SBOM after Black Duck / Anchore scans, or `syft` against the built `full` Docker image.

## Regenerate

```bash
./scripts/generate_sbom.sh
```

CI runs the same script via `.github/workflows/sbom.yml` (manual dispatch, or when dependency manifests change on `main`) and uploads the artifacts.
