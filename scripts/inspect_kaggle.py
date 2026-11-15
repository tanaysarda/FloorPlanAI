from pathlib import Path
import cv2
import kagglehub
import numpy as np

# 1. Download/locate dataset in kagglehub cache
dataset_path = Path(kagglehub.dataset_download("asutoshprad/floor-plan-dataset"))
print("Dataset path:", dataset_path)

# 2. Print folder layout & sample filenames
all_files = sorted(p for p in dataset_path.rglob("*") if p.is_file())
print(f"\nTotal files: {len(all_files)}")
for p in all_files[:25]:
    print(" ", p.relative_to(dataset_path))

# 3. Inspect pixel values/colors of the first PNG mask found
png_files = [p for p in all_files if p.suffix.lower() == ".png"]
if png_files:
    sample = cv2.imread(str(png_files[0]), cv2.IMREAD_UNCHANGED)
    print(f"\nSample file ({png_files[0].name}) shape: {sample.shape}, dtype: {sample.dtype}")
    if sample.ndim == 2:
        print("Unique label IDs:", np.unique(sample))
    else:
        # Convert BGR to RGB and list top unique colors
        rgb = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
        colors, counts = np.unique(rgb.reshape(-1, 3), axis=0, return_counts=True)
        order = np.argsort(-counts)
        print("Top RGB colors (R, G, B) -> pixel count:")
        for c, cnt in zip(colors[order[:15]], counts[order[:15]]):
            print(f"  {c.tolist()} -> {cnt}")