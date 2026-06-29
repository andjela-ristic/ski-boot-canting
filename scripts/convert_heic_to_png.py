from pathlib import Path
from PIL import Image
from pillow_heif import register_heif_opener
import json
import csv


register_heif_opener()


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw_originals"
OUTPUT_DIR = PROJECT_ROOT / "data" / "working_png"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"

CSV_PATH = METADATA_DIR / "conversion_log.csv"
JSON_PATH = METADATA_DIR / "conversion_log.json"


def convert_heic_to_png() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    heic_files = sorted(
        list(RAW_DIR.glob("*.HEIC")) +
        list(RAW_DIR.glob("*.heic"))
    )

    if not heic_files:
        print(f"No HEIC files found in: {RAW_DIR}")
        return

    metadata = []

    for heic_path in heic_files:
        output_path = OUTPUT_DIR / f"{heic_path.stem}.png"

        if output_path.exists():
            print(f"Skipping existing file: {output_path.name}")
            continue

        try:
            with Image.open(heic_path) as img:
                original_mode = img.mode
                original_size = img.size

                # PNG/OpenCV-friendly format
                img = img.convert("RGB")
                img.save(output_path, format="PNG")

            record = {
                "source_file": str(heic_path.relative_to(PROJECT_ROOT)),
                "output_file": str(output_path.relative_to(PROJECT_ROOT)),
                "original_width": original_size[0],
                "original_height": original_size[1],
                "original_mode": original_mode,
                "converted_mode": "RGB",
            }

            metadata.append(record)

            print(f"Converted: {heic_path.name} -> {output_path.name}")

        except Exception as e:
            print(f"Failed to convert {heic_path.name}: {e}")

            metadata.append({
                "source_file": str(heic_path.relative_to(PROJECT_ROOT)),
                "output_file": None,
                "error": str(e),
            })

    save_metadata(metadata)

    print()
    print(f"Done. Converted files are in: {OUTPUT_DIR}")
    print(f"Metadata saved to: {CSV_PATH}")
    print(f"Metadata saved to: {JSON_PATH}")


def save_metadata(metadata: list[dict]) -> None:
    if not metadata:
        return

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    fieldnames = sorted({key for row in metadata for key in row.keys()})

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata)


if __name__ == "__main__":
    convert_heic_to_png()