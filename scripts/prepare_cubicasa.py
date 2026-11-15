from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import kagglehub
import numpy as np

# Integer class IDs matching your class map
CLASS_IDS = {
    "Background": 0,
    "Wall": 1,
    "Door": 2,
    "Window": 3,
    "Bedroom": 4,
    "Kitchen": 5,
    "LivingRoom": 6,
    "Bath": 7,
    "Hallway": 8,
    "Room": 9,  # fallback for other room polygons
}


def parse_points(points_str: str) -> np.ndarray:
    pts = []
    for pair in points_str.strip().split():
        if "," in pair:
            x, y = pair.split(",")[:2]
            pts.append([int(round(float(x))), int(round(float(y)))])
    return np.array(pts, dtype=np.int32)


def svg_to_mask(svg_path: Path, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    tree = ET.parse(svg_path)
    root = tree.getroot()

    # Draw rooms first, then walls, then windows/doors on top
    draw_order = ["Room", "Wall", "Window", "Door"]
    polygons_by_layer = {k: [] for k in draw_order}

    for elem in root.iter():
        tag = elem.tag.split("}")[-1]
        if tag != "polygon":
            continue
        cls_attr = elem.attrib.get("class", "")
        pts_attr = elem.attrib.get("points", "")
        if not pts_attr:
            continue
        pts = parse_points(pts_attr)
        if len(pts) < 3:
            continue

        if "Wall" in cls_attr:
            polygons_by_layer["Wall"].append((CLASS_IDS["Wall"], pts))
        elif "Door" in cls_attr:
            polygons_by_layer["Door"].append((CLASS_IDS["Door"], pts))
        elif "Window" in cls_attr:
            polygons_by_layer["Window"].append((CLASS_IDS["Window"], pts))
        elif "Space" in cls_attr:
            cid = CLASS_IDS["Room"]
            for room_name, rid in CLASS_IDS.items():
                if room_name in cls_attr:
                    cid = rid
                    break
            polygons_by_layer["Room"].append((cid, pts))

    for layer in draw_order:
        for cid, pts in polygons_by_layer[layer]:
            cv2.fillPoly(mask, [pts], int(cid))

    return mask


def prepare_from_kagglehub(
    out_images: Path = Path("data/kaggle/images"),
    out_masks: Path = Path("data/kaggle/masks"),
    limit: int = 20,
) -> None:
    # Resolves to your local kagglehub cache path (reuses cache on subsequent runs)
    dataset_path = Path(kagglehub.dataset_download("qmarva/cubicasa5k"))
    print("Using CubiCasa5K dataset at:", dataset_path)

    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    plan_dirs = sorted(p.parent for p in dataset_path.rglob("model.svg"))[:limit]
    if not plan_dirs:
        raise FileNotFoundError(f"No model.svg files found under {dataset_path}")

    for idx, pdir in enumerate(plan_dirs, start=1):
        img_file = pdir / "F1_scaled.png"
        if not img_file.exists():
            img_file = pdir / "F1_original.png"
        svg_file = pdir / "model.svg"
        if not img_file.exists() or not svg_file.exists():
            continue

        img = cv2.imread(str(img_file))
        if img is None:
            continue
        h, w = img.shape[:2]
        mask = svg_to_mask(svg_file, h, w)

        stem = f"cubicasa_{idx:03d}_{pdir.name}"
        cv2.imwrite(str(out_images / f"{stem}.png"), img)
        cv2.imwrite(str(out_masks / f"{stem}.png"), mask)
        print(f"Prepared {stem}")


if __name__ == "__main__":
    prepare_from_kagglehub(limit=20)