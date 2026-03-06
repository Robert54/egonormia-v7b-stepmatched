# Data Note

This repo does not include training data or videos.

Expected training annotation files:
- `egonormia_llava_v6_train.json`
- `egonormia_llava_v6b_train.json`
- `egonormia_llava_v7_cot_mcq3_train.json`

Expected evaluation files:
- `egonormia_llava_test.json`
- `final_data.json`

Expected media root:
- EgoNormia video directory containing sample-relative paths referenced by the JSON

Expected annotation structure:
- LLaVA-style records
- `video` field pointing to the video asset
- `conversations` field used by `llava_sft.py`

The original runs used:
- `v6`: natural-language-output training JSON
- `v6b`: natural-language-output plus sensibility-generation training JSON
- `v7b-stepmatched`: 4890 MCQ records
- `v7b-stepmatched`: 1630 action + 1630 justification + 1630 sensibility samples
