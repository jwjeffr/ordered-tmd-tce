from pathlib import Path
from typing import Iterable, Tuple

from PIL import Image, ImageOps


def save_image_grid(
    image_dir: Path,
    grid_size: Tuple[int, int],
    outline_width: int,
    output_path: Path = Path("grid.png"),
) -> Path:
    """
    Build a grid image from files in `image_dir` and save it to `output_path`.

    Args:
        image_dir: Directory containing image files.
        grid_size: (cols, rows)
        outline_width: Black border width around each image, in pixels.
        output_path: Where to save the final montage.

    Returns:
        The path to the saved grid image.
    """
    image_dir = Path(image_dir)
    output_path = Path(output_path)
    cols, rows = grid_size

    if cols <= 0 or rows <= 0:
        raise ValueError("grid_size must be positive, e.g. (3, 4)")
    if outline_width < 0:
        raise ValueError("outline_width must be >= 0")

    # Load image files from the directory.
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    paths = sorted(
        (p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in valid_exts),
        key=lambda path: int(path.stem)
    )

    if not paths:
        raise ValueError(f"No image files found in {image_dir}")

    images = [Image.open(p).convert("RGB") for p in paths[: cols * rows]]

    # Compute a uniform inner cell size so every tile aligns nicely.
    max_w = max(img.width for img in images)
    max_h = max(img.height for img in images)
    cell_w = max_w + 2 * outline_width
    cell_h = max_h + 2 * outline_width

    grid_img = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")

    for i in range(cols * rows):
        row, col = divmod(i, cols)
        x0 = col * cell_w
        y0 = row * cell_h

        if i >= len(images):
            continue  # Leave remaining cells blank white.

        img = images[i]

        # Fit image inside the cell while preserving aspect ratio.
        fitted = ImageOps.contain(img, (max_w, max_h), method=Image.Resampling.LANCZOS)

        # Add the black outline around the fitted image.
        bordered = ImageOps.expand(fitted, border=outline_width, fill="black")

        # Center the bordered image within its cell.
        paste_x = x0 + (cell_w - bordered.width) // 2
        paste_y = y0 + (cell_h - bordered.height) // 2
        grid_img.paste(bordered, (paste_x, paste_y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid_img.save(output_path)
    return output_path


if __name__ == "__main__":
    out = save_image_grid(
        image_dir=Path("images"),
        grid_size=(4, 5),
        outline_width=6,
        output_path=Path("grid.png"),
    )
    print(f"Saved to {out}")