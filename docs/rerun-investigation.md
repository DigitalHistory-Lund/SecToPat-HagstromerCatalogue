# Investigation: Did the pipeline rerun produce meaningfully different results?

**Conclusion: Yes.** The rerun changed card content significantly in many
files — not just OCR noise. Some slots now contain entirely different
physical cards, indicating the card detector re-sliced pages differently.

---

## 1. Scope of changes

The rerun commit (`44396e8 Rerun whole pipeline, replace all`) was
compared against its parent:

```bash
git diff --stat 44396e8^..44396e8 -- transcriptions/ | tail -5
```

```
4288 files changed, 12954 insertions(+), 12454 deletions(-)
```

Net +500 lines of content across 4,288 of the 5,492 transcription files.

## 2. Were cards added or removed?

```bash
git diff 44396e8^..44396e8 --diff-filter=A --name-only \
  -- transcriptions/ | wc -l
# 0
git diff 44396e8^..44396e8 --diff-filter=D --name-only \
  -- transcriptions/ | wc -l
# 0
```

Same for `extracted_cards/` — zero files added or deleted. The total
card count is unchanged; what changed is the *content* mapped to each
slot.

## 3. How large are the changes?

```bash
git diff --numstat 44396e8^..44396e8 -- transcriptions/ \
  | awk '{d=$1+$2; if(d<=2) tiny++; else if(d<=6) small++; \
    else if(d<=20) medium++; else big++} \
    END {print "Tiny (<=2):", tiny; print "Small (3-6):", small; \
         print "Medium (7-20):", medium; print "Large (>20):", big}'
```

```
Tiny (<=2 lines changed): 1,167
Small (3-6):              1,860
Medium (7-20):            1,162
Large (>20):                 99
```

Over 1,200 files have 7+ lines changed — well beyond minor OCR jitter.

## 4. Which files changed most?

```bash
git diff --numstat 44396e8^..44396e8 -- transcriptions/ \
  | sort -rn | head -10
```

```
38  37  transcriptions/09_0000_1_2.txt
37  16  transcriptions/09_0001_0_2.txt
36  35  transcriptions/09_0001_1_1.txt
34  36  transcriptions/09_0002_0_1.txt
32  33  transcriptions/09_0002_1_2.txt
30  30  transcriptions/09_0001_1_2.txt
22  25  transcriptions/09_0002_0_3.txt
19  11  transcriptions/01_0102_0_2.txt
17   9  transcriptions/10_0000_0_3.txt
17  11  transcriptions/01_0102_1_2.txt
```

Volume 09 dominates the top — its first few pages were re-sliced most
aggressively.

## 5. Spot-check: genuinely different cards

```bash
git diff 44396e8^..44396e8 -- transcriptions/09_0001_0_2.txt
```

The old version listed correspondents from "England, Australien, New
Zealand" while the new version lists "U.S.A. och Canada" — a completely
different physical card, not an OCR variation. This confirms the card
detector assigned different page regions to the same filename slot.

## Summary

| Finding                        | Detail                                |
| ------------------------------ | ------------------------------------- |
| Files changed                  | 4,288 of 5,492 (78%)                 |
| Files added/removed            | 0 — same card count                  |
| Net content change             | +500 lines                            |
| Meaningful changes (>6 lines)  | 1,261 files                           |
| Most affected volume           | 09 (top 7 largest diffs)              |
| Nature of large changes        | Different physical cards in same slots |
