# Hardware feasibility

```json
{
  "platform": "macOS-26.5.1-arm64-arm-64bit",
  "torch": "2.14.1",
  "transformers": "4.57.6",
  "mps_available": true,
  "device": "mps",
  "model": "answerdotai/ModernBERT-base",
  "memory_bytes": 25769803776,
  "probes": [
    {
      "length": 256,
      "batch": 1,
      "steps": 3,
      "seconds": 1.5529302089998964,
      "first_loss": 1.8059406280517578,
      "last_loss": 1.489651083946228,
      "allocated_bytes": 2421331712,
      "driver_bytes": 3735781376
    },
    {
      "length": 512,
      "batch": 1,
      "steps": 100,
      "seconds": 25.178405999999995,
      "first_loss": 1.3964179754257202,
      "last_loss": 0.8134176731109619,
      "allocated_bytes": 2422908672,
      "driver_bytes": 4843110400
    },
    {
      "length": 1024,
      "batch": 1,
      "steps": 3,
      "seconds": 1.6155417090003539,
      "first_loss": 0.820650577545166,
      "last_loss": 0.8187028765678406,
      "allocated_bytes": 2427242240,
      "driver_bytes": 5916884992
    }
  ],
  "passed": true,
  "model_revision": "8949b909ec900327062f0ebf497f51aef5e6f0c8",
  "parameters": 149014272,
  "bf16": true,
  "gradient_checkpointing": true
}
```
