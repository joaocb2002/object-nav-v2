# Reproducibility tools

These commands check local experiment contracts without becoming part of the
interactive navigation loop.

`check_yolo_contract.py` loads the local YOLO11x weights and a stored RGB
fixture. Its default `both` mode starts separate Python processes for stock
Ultralytics and the repository softmax patch, preventing the process-global
patch from contaminating the stock result.

```bash
python3 scripts/tools/check_yolo_contract.py --mode both --device cpu
```

The check requires Ultralytics 8.4.82 and
`models/yolo/yolo11x.pt`. See `docs/yolo_contract.md` for the expected result
shapes, weight checksum, and patch limitations.
