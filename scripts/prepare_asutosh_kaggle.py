from pathlib import Path
import shutil
import kagglehub

dataset_path = Path(kagglehub.dataset_download("asutoshprad/floor-plan-dataset"))
x_dir = dataset_path / "X"
y_dir = dataset_path / "Y"

out_images = Path("data/kaggle/images")
out_masks = Path("data/kaggle/masks")
out_images.mkdir(parents=True, exist_ok=True)
out_masks.mkdir(parents=True, exist_ok=True)

x_files = sorted(x_dir.glob("*.png"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)

count = 0
for img_path in x_files:
    mask_path = y_dir / f"{img_path.stem} (2).png"
    if not mask_path.exists():
        continue
    count += 1
    stem = f"kaggle_plan_{count:03d}"
    shutil.copy2(img_path, out_images / f"{stem}.png")
    shutil.copy2(mask_path, out_masks / f"{stem}.png")
    if count >= 25:
        break

print(f"Prepared {count} paired images and masks in data/kaggle/")