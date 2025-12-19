# mahjong-helper MJAI bot

This bot delegates discard and call recommendations to the external
`mahjong-helper` CLI engine (EndlessCheng/mahjong-helper) while keeping
Akagi's UI and MITM bridge intact.

## Usage

- Ensure `mahjong-helper` is on `PATH`, or set `MAHJONG_HELPER_BIN` to the
  binary path.
- Optional timeout override: `MAHJONG_HELPER_TIMEOUT_MS` (default: 1500ms).
- Select `mahjong_helper` as the model in Akagi's UI.

## Notes

- Melds are passed via `#` when known; concealed kans are uppercased.
- Dora indicators are converted to actual dora tiles and passed via `-d=...`.
- Call decisions are best-effort; the bot passes when output is uncertain.
- Recommendations are mapped to Akagi's UI via `meta` fields.
- Auto-riichi when tenpai and auto-nukidora in 3P when a north tile is held.

## Self-test

Run the parser tests (no MJAI dependency):

```bash
python Akagi/mjai_bot/mahjong_helper/self_test.py
```
