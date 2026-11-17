# Unpack the match data here

This folder is empty on purpose. The match assets are gated and must not be
committed. See the official `DATA.md` for enrolment and the download link.

Unpack the official bundle so that the files keep the names they arrive with:

```
data/
└─ tactical/
   ├─ BAR-ATH_tactical_atd.xml
   ├─ BAR-ATH_tactical_atd.txt
   └─ BAR-ATH_tactical_smart_tagging.xml
```

The pipeline uses only those three tactical files. The tactical video,
homography, and the broadcast recording are not inputs.

You may also keep the bundle elsewhere and pass its root to `--data`, or set
`MTAH_DATA_DIR` to that root. The root is the folder that contains `tactical/`.
