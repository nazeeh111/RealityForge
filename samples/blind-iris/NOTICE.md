# Held-out IRIS recording subset

These six transformed recordings are from **IRIS**, by Rasul Khanbayov, Mohamed Rayan Barhdadi, Erchin Serpedin and Hasan Kurban.

Source: https://huggingface.co/datasets/rasulkhanbayov/IRIS , revision `c253822f55431ca80ef2084de4bc5e79d1a488f1`.

License: **Creative Commons Attribution-NonCommercial 4.0 International**, https://creativecommons.org/licenses/by-nc/4.0/ . The application's MIT license does not relicense these recordings. Their use is restricted to noncommercial purposes under that license.

Changes: fixed source2..10second intervals from take03 and take06 of pendulum20,45,90; resized to960x540, reduced60fps to30fps without changing playback speed, audio removed, H.264 encoding. Exact source URLs, selection protocol, transformations, first-frame-only tracking regions and hashes are in `../blind-provenance.json`.

The estimator source and model hashes were frozen before acquiring or inspecting these six recordings. All attempted recordings and failures are retained in `../../benchmarks/blind-iris.json`. The subset tests recording transfer after development; it is not the complete official IRIS protocol. No IRIS recording is used in neural-network training.

Reference: Khanbayov et al., *IRIS: A Real-World Benchmark for Inverse Recovery and Identification of Physical Dynamic Systems from Monocular Video*, ECCV2026 / arXiv:2603.16432, https://arxiv.org/abs/2603.16432 .
