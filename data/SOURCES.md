# Data sources

## Official data, obtained separately

Publisher: COMAP. Filename: `2026_MCM_Problem_C_Data.csv`.

- [Official problem list](https://www.contest.comap.com/undergraduate/contests/mcm/contests/2026/problems/)
- [Official CSV](https://www.contest.comap.com/undergraduate/contests/mcm/contests/2026/problems/2026_MCM_Problem_C_Data.csv)
- [Official problem description](https://www.contest.comap.org/undergraduate/contests/mcm/contests/2026/problems/2026_MCM_Problem_C.pdf)

The audited source was retrieved on 2026-10-12: 90,002 bytes, 421 rows and 53 columns.
SHA-256: `ea99caec6ea243bdb450a1971a95ba8a95701a93be7ff29f0ba3c57d72ddff52`.

The CSV is not bundled. COMAP retains applicable rights; public availability is not
represented as an open-data license. Consult the publisher's terms for intended use.
`python scripts/download_data.py` explicitly requests the source, verifies the checksum
before writing, and refuses overwrites. A changed or inaccessible upstream file is a
blocker to investigate, not permission to quietly substitute another dataset.

## Synthetic demo

`dwts/demo.py` defines four invented contestants and invented fan shares. These inputs
have no real contestant identities and are not statistical evidence about DWTS.

## Source report

The report and extracted text are not included. See [PROVENANCE.md](../docs/PROVENANCE.md)
for attribution and the report checksum. No original-team code was supplied.
