---
language:
- de
license: apache-2.0
tags:
- voice
- classification
- age
- gender
- speech
- audio
datasets:
- mozilla-foundation/common_voice_12_0
widget:
- src: >-
    https://huggingface.co/padmalcom/wav2vec2-asr-ultimate-german/resolve/main/test.wav
  example_title: Sample 1
pipeline_tag: audio-classification
metrics:
- accuracy
---

This multi-task wav2vec2 based asr model has two additional classification heads to detect:
- age
- gender
... of the current speaker in one forward pass.

It was trained on  [mozilla common voice](https://commonvoice.mozilla.org/).

Code for training can be found [here](https://github.com/padmalcom/wav2vec2-asr-ultimate-german).

*inference.py* shows, how the model can be used.