<!-- RECOMMENDATION — for the user to confirm before publishing. Not yet a binding license. -->
## License (recommended — please confirm)

- **Robot dataset** (`banana_in_pot_*` on HF): recommend **CC-BY-NC 4.0** per REPRODUCIBILITY_PLAN §9-R4 — it contains real lab-workspace video, so a non-commercial, attribution license bounds that exposure while allowing research reuse. Confirm the lab is OK with the data already being public before finalizing.
- **HF dataset cards** (`Bigenlight/banana_in_pot_lerobot_v3`, `banana_in_pot_raw`, `banana_in_pot_ee_lerobot_v3`, `act_banana_in_pot`) should carry the **same CC-BY-NC 4.0** so the license travels with the data.
- **Code in this repo** (scripts, converters, docs): a permissive **MIT** or **Apache-2.0** is the usual fit — decide separately from the dataset license.
- **Third-party**: `lerobot` (Apache-2.0) and `gello_software` keep their own upstream licenses; this note does not relicense them.

> Action for the user: pick and drop a top-level `LICENSE` file + a one-line license line in `README.md`, and mirror the dataset license onto every HF card.
