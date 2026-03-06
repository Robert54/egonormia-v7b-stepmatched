# Data Note

This repo does not include training data or videos.

Expected training annotation file:
- `egonormia_llava_v7_cot_mcq3_train.json`

Expected media root:
- EgoNormia video directory containing sample-relative paths referenced by the JSON

Expected annotation structure:
- LLaVA-style records
- `video` field pointing to the video asset
- `conversations` field used by `llava_sft.py`

The original mainline run used:
- 4890 MCQ records
- 1630 action samples
- 1630 justification samples
- 1630 sensibility samples
