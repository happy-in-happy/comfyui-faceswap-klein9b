"""
Happyin Contact Sheet — combines images into a labeled grid.

Two modes (can be combined):
  1. Fixed slots: image_1..image_5 with label_1..label_5 for standardized comparisons
  2. Dynamic batch: images + labels for CameraAngles-style workflows

Fixed slot images appear first, then batch images.
Connect CameraAngles `labels` output -> labels input for rich annotations.
Connect CameraAngles `debug` output -> title for preset/face detection info.

Optional `prompt` input renders as a separate text block in the grid.
Cols = "auto" by default: picks optimal column count based on image count.
"""

import os
import torch
import numpy as np
import textwrap
import math

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None


def _get_font(size, mono=False):
    """Get a font. Prefers sans-serif for labels, mono for code/debug."""
    if Image is None:
        return None
    # Try models mount first (Docker containers with WekaFS)
    # Path: happyin-comfyui-nodes -> custom_nodes -> ComfyUI -> models
    _models_font = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "models", "_fonts_DejaVuSans.ttf")

    if mono:
        families = [
            _models_font,  # WekaFS mount fallback
            # Linux
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/custom/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
            "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
            # Windows
            "C:/Windows/Fonts/consola.ttf",
            "C:/Windows/Fonts/cour.ttf",
        ]
    else:
        families = [
            _models_font,  # WekaFS mount fallback
            # Linux
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/custom/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
            # Windows
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
        ]
    try:
        for path in families:
            try:
                return ImageFont.truetype(path, size)
            except (OSError, IOError):
                continue
        return ImageFont.load_default()
    except Exception:
        return ImageFont.load_default()


def _measure_text_height(draw, text, font, max_width):
    """Measure height needed for text that may wrap."""
    lines = text.split('\n')
    total_h = 0
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_w = bbox[2] - bbox[0]
        line_h = bbox[3] - bbox[1]
        if line_w > max_width and max_width > 0:
            chars_per_line = max(1, int(len(line) * max_width / max(line_w, 1)))
            wrapped = textwrap.wrap(line, width=chars_per_line)
            total_h += line_h * max(len(wrapped), 1)
        else:
            total_h += line_h
    return max(total_h, 12)


def _auto_cols(n):
    """Pick optimal column count based on number of images.

    1       -> 1
    2       -> 2
    3       -> 3
    4       -> 2  (2x2)
    5-6     -> 3  (2x3)
    7-9     -> 3  (3x3)
    10-12   -> 4  (3x4)
    13+     -> ceil(sqrt(n))
    """
    if n <= 3:
        return n
    if n == 4:
        return 2
    if n <= 9:
        return 3
    if n <= 12:
        return 4
    return math.ceil(math.sqrt(n))


def _render_prompt_block(text, width, height):
    """Render prompt text into a PIL image that fits within width x height.

    Auto-sizes font: starts large, shrinks until all text fits.
    Returns a PIL Image of exactly (width, height).
    """
    img = Image.new("RGB", (width, height), (30, 30, 38))
    draw = ImageDraw.Draw(img)

    pad = 16
    usable_w = width - pad * 2
    usable_h = height - pad * 2

    if not text or not text.strip():
        return img

    text = text.strip()

    # Auto-size: start from large font, shrink until it fits
    for fsize in range(48, 9, -2):
        font = _get_font(fsize)
        # Estimate chars per line
        bbox_test = draw.textbbox((0, 0), "W" * 10, font=font)
        char_w = (bbox_test[2] - bbox_test[0]) / 10.0
        chars_per_line = max(1, int(usable_w / char_w))
        wrapped_lines = []
        for raw_line in text.split('\n'):
            if raw_line.strip() == '':
                wrapped_lines.append('')
            else:
                wrapped_lines.extend(textwrap.wrap(raw_line, width=chars_per_line) or [''])

        # Measure total height
        line_bbox = draw.textbbox((0, 0), "Ay", font=font)
        line_h = (line_bbox[3] - line_bbox[1]) + 4
        total_h = line_h * len(wrapped_lines)

        if total_h <= usable_h:
            # Fits! Draw it centered vertically
            y_start = pad + (usable_h - total_h) // 2
            for i, line in enumerate(wrapped_lines):
                lbbox = draw.textbbox((0, 0), line, font=font)
                lw = lbbox[2] - lbbox[0]
                x = pad + (usable_w - lw) // 2
                y = y_start + i * line_h
                draw.text((x, y), line, fill=(220, 220, 210), font=font)
            break
    else:
        # Even smallest font does not fit -- draw truncated with smallest
        font = _get_font(10)
        draw.text((pad, pad), text[:500], fill=(220, 220, 210), font=font)

    # Subtle border
    draw.rectangle([(0, 0), (width - 1, height - 1)], outline=(70, 70, 80), width=1)

    return img


class HappyinContactSheet:
    """Combines images into a labeled contact sheet grid.

    Two modes (combinable):
      Fixed slots: image_1..image_5 + label_1..label_5 for standardized comparison.
      Dynamic batch: images + labels for CameraAngles-style workflows.

    Fixed slots appear first on the grid, then batch images.
    Optional prompt input is rendered as a separate text block at the end.
    Cols=0 means auto-detect optimal column count.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cols": ("INT", {"default": 0, "min": 0, "max": 10,
                    "tooltip": "Колонки. 0 = авто (подбирает по кол-ву фото)"}),
            },
            "optional": {
                # -- Fixed slots (standardized labels) --------
                "image_1": ("IMAGE", {"tooltip": "Fixed slot 1"}),
                "label_1": ("STRING", {"default": "",
                    "tooltip": "Label for image_1 (empty = '#1')"}),
                "image_2": ("IMAGE", {"tooltip": "Fixed slot 2"}),
                "label_2": ("STRING", {"default": "",
                    "tooltip": "Label for image_2 (empty = '#2')"}),
                "image_3": ("IMAGE", {"tooltip": "Fixed slot 3"}),
                "label_3": ("STRING", {"default": "",
                    "tooltip": "Label for image_3 (empty = '#3')"}),
                "image_4": ("IMAGE", {"tooltip": "Fixed slot 4"}),
                "label_4": ("STRING", {"default": "",
                    "tooltip": "Label for image_4 (empty = '#4')"}),
                "image_5": ("IMAGE", {"tooltip": "Fixed slot 5"}),
                "label_5": ("STRING", {"default": "",
                    "tooltip": "Label for image_5 (empty = '#5')"}),
                # -- Dynamic batch ----------------------------
                "images": ("IMAGE", {
                    "tooltip": "Batch — подключи батч картинок. "
                               "Каждая картинка в батче = отдельная ячейка."}),
                "labels": ("STRING", {"default": "", "multiline": True,
                    "tooltip": "Лейблы для батча (по одному на строку).\n"
                               "Можно подключить от CameraAngles."}),
                "title": ("STRING", {"forceInput": True,
                    "tooltip": "Title (from CameraAngles `debug`)."}),
                # -- Prompt text block ------------------------
                "prompt": ("STRING", {"forceInput": True,
                    "tooltip": "Prompt text -- displayed as a separate block in the grid."}),
            },
        }

    INPUT_IS_LIST = True

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("contact_sheet",)
    FUNCTION = "make_sheet"
    CATEGORY = "Happyin/Camera"

    DESCRIPTION = (
        "Contact Sheet -- grid of images with labels.\n\n"
        "Fixed slots: image_1..5 + label_1..5 for comparisons.\n"
        "Dynamic mode: images + labels (from CameraAngles).\n"
        "Both modes combine: fixed first, then batch.\n\n"
        "prompt: rendered as a separate text block.\n"
        "cols=0: auto-detect optimal column count.\n"
        "title: header (from CameraAngles `debug`)."
    )

    @staticmethod
    def _unwrap_scalar(val, default=None):
        """Unwrap INPUT_IS_LIST scalar: [x] -> x."""
        if isinstance(val, list) and len(val) > 0:
            return val[0]
        if val is None:
            return default
        return val

    @staticmethod
    def _unwrap_images(val):
        """Unwrap INPUT_IS_LIST images into flat list of 3D tensors."""
        if val is None:
            return []
        items = val if isinstance(val, list) else [val]
        out = []
        for img in items:
            if isinstance(img, torch.Tensor):
                if img.dim() == 4:
                    for i in range(img.shape[0]):
                        out.append(img[i])
                elif img.dim() == 3:
                    out.append(img)
            else:
                if img is not None:
                    out.append(img)
        return out

    def make_sheet(self, cols,
                   image_1=None, label_1=None,
                   image_2=None, label_2=None,
                   image_3=None, label_3=None,
                   image_4=None, label_4=None,
                   image_5=None, label_5=None,
                   images=None, labels=None, title=None,
                   prompt=None):
        if Image is None:
            print("[HappyinContactSheet] PIL not available")
            return (torch.zeros(1, 64, 64, 3),)

        # Unwrap scalar inputs (INPUT_IS_LIST wraps everything in lists)
        cols_val = self._unwrap_scalar(cols, 0)
        title_val = self._unwrap_scalar(title, "")
        prompt_val = self._unwrap_scalar(prompt, "")

        # -- Collect fixed-slot images + labels ---------------
        img_list = []
        label_list = []

        slot_images = [image_1, image_2, image_3, image_4, image_5]
        slot_labels = [label_1, label_2, label_3, label_4, label_5]

        for idx, (s_img, s_lbl) in enumerate(zip(slot_images, slot_labels), 1):
            slot_imgs = self._unwrap_images(s_img)
            if not slot_imgs:
                continue
            lbl = self._unwrap_scalar(s_lbl, "")
            if not lbl:
                lbl = f"#{idx}"
            for j, t in enumerate(slot_imgs):
                img_list.append(t)
                if len(slot_imgs) > 1:
                    label_list.append(f"{lbl} [{j + 1}]")
                else:
                    label_list.append(lbl)

        # -- Collect dynamic batch images + labels ------------
        batch_imgs = self._unwrap_images(images)

        # Parse labels: supports both connected list ["a","b","c"]
        # and typed multiline text ["a\nb\nc"] (INPUT_IS_LIST wraps both)
        if labels is None:
            batch_labels = []
        elif isinstance(labels, list):
            batch_labels = []
            for lb in labels:
                if not isinstance(lb, str):
                    lb = str(lb)
                lb = lb.replace("<sks> ", "")
                # If single string with newlines → split into individual labels
                if '\n' in lb:
                    batch_labels.extend(
                        line.strip() for line in lb.split('\n')
                        if line.strip())
                elif lb.strip():
                    batch_labels.append(lb.strip())
        else:
            s = str(labels).replace("<sks> ", "")
            if '\n' in s:
                batch_labels = [l.strip() for l in s.split('\n') if l.strip()]
            elif s.strip():
                batch_labels = [s.strip()]
            else:
                batch_labels = []

        # Fill missing labels with auto-numbers
        while len(batch_labels) < len(batch_imgs):
            batch_labels.append(f"#{len(img_list) + len(batch_labels) + 1}")
        batch_labels = batch_labels[:len(batch_imgs)]

        img_list.extend(batch_imgs)
        label_list.extend(batch_labels)

        # Filter out 1x1 black placeholders (from ImagePick "no real image found")
        real_pairs = [(t, l) for t, l in zip(img_list, label_list)
                      if isinstance(t, torch.Tensor) and t.shape[0] > 1 and t.shape[1] > 1]
        if real_pairs:
            img_list, label_list = zip(*real_pairs)
            img_list, label_list = list(img_list), list(label_list)
        else:
            img_list, label_list = [], []

        n = len(img_list)
        if n == 0:
            blank = torch.zeros(1, 64, 64, 3)
            return (blank,)

        # Convert tensors to PIL images
        pil_images = []
        for t in img_list:
            if not isinstance(t, torch.Tensor):
                continue
            arr = t.cpu().numpy()
            if arr is None or not hasattr(arr, 'dtype') or arr.dtype == object:
                continue
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            pil_images.append(Image.fromarray(arr))

        if not pil_images:
            blank = torch.zeros(1, 64, 64, 3)
            return (blank,)

        # Sync label_list to pil_images count (in case conversion skipped any)
        n = len(pil_images)
        label_list = label_list[:n]
        while len(label_list) < n:
            label_list.append(f"#{len(label_list) + 1}")

        # No resizing — keep original image sizes

        # -- Prompt: will be rendered as footer (not a grid item) --
        has_prompt = bool(prompt_val and prompt_val.strip())

        # -- Cols logic -----------------------------------------
        auto_cols = cols_val <= 0
        if auto_cols:
            cols_val = _auto_cols(n)

        # Font size dynamic — proportional to each image's WIDTH
        # For layout (label_h) use the max font across all images
        max_h = max(img.height for img in pil_images)
        max_w = max(img.width for img in pil_images)
        max_font = min(max(max_w // 20, 14), 120)
        gap = max(10, max_h // 40)  # proportional gap between images

        # Measure label height using the largest possible font
        temp_img = Image.new("RGB", (1, 1))
        temp_draw = ImageDraw.Draw(temp_img)
        layout_font = _get_font(max_font)
        sample_bbox = temp_draw.textbbox((0, 0), "Aygjpq|", font=layout_font)
        single_line_h = sample_bbox[3] - sample_bbox[1]
        label_h = single_line_h + max_font // 3

        # Title — use font proportional to average width
        avg_w = sum(img.width for img in pil_images) // max(n, 1)
        title_font_size = min(max(avg_w // 20, 14), 120)
        title_h = 0
        if title_val:
            title_h = title_font_size + title_font_size // 2

        # Build rows
        all_items = list(zip(pil_images, label_list))
        rows_layout = []

        if auto_cols:
            # Dynamic packing: wrap when width exceeds limit
            avg_w = sum(img.width for img in pil_images) // max(n, 1)
            max_row_width = avg_w * cols_val
            current_row = []
            current_width = 0
            for item in all_items:
                img_w = item[0].width
                needed = img_w + (gap if current_row else 0)
                if current_row and current_width + needed > max_row_width:
                    rows_layout.append(current_row)
                    current_row = []
                    current_width = 0
                    needed = img_w
                current_row.append(item)
                current_width += needed
            if current_row:
                rows_layout.append(current_row)
        else:
            # Fixed cols: exactly cols_val images per row
            for i in range(0, n, cols_val):
                rows_layout.append(all_items[i:i + cols_val])

        nrows = len(rows_layout)
        row_widths = [
            sum(img.width for img, _ in row) + gap * (len(row) - 1)
            for row in rows_layout
        ]
        row_heights = [max(img.height for img, _ in row) for row in rows_layout]
        sheet_w = max(row_widths)

        # Each row height = tallest image in that row + label
        row_y_offsets = []
        y = title_h
        for rh in row_heights:
            row_y_offsets.append(y)
            y += rh + label_h
        grid_h = y

        # -- Prompt footer: multi-column text (matches image grid cols) ---
        prompt_footer_h = 0
        prompt_font = None
        prompt_columns = []   # list of lists of lines
        prompt_col_count = 0
        prompt_pad = 0
        prompt_line_h = 0
        prompt_col_w = 0
        prompt_col_gap = 0
        if has_prompt:
            # Font: proportional to sheet width
            pfont_size = min(max(sheet_w // 45, 18), 72)
            prompt_font = _get_font(pfont_size, mono=False)
            prompt_pad = max(24, pfont_size // 2)
            # Text columns = image columns (fallback 1)
            prompt_col_count = max(cols_val, 1)
            prompt_col_gap = gap
            total_gaps = prompt_col_gap * max(prompt_col_count - 1, 0)
            prompt_col_w = (sheet_w - prompt_pad * 2 - total_gaps) // max(
                prompt_col_count, 1)
            # Wrap text to column width
            tmp = Image.new("RGB", (1, 1))
            tmp_d = ImageDraw.Draw(tmp)
            char_bbox = tmp_d.textbbox((0, 0), "W" * 10, font=prompt_font)
            char_w = (char_bbox[2] - char_bbox[0]) / 10.0
            chars_per_line = max(1, int(prompt_col_w / max(char_w, 1)))
            all_lines = []
            for raw_line in prompt_val.strip().split('\n'):
                if raw_line.strip() == '':
                    all_lines.append('')
                else:
                    all_lines.extend(
                        textwrap.wrap(raw_line, width=chars_per_line) or [''])
            # Distribute lines evenly across columns
            lines_per_col = math.ceil(len(all_lines) / max(prompt_col_count, 1))
            prompt_columns = []
            for c in range(prompt_col_count):
                start = c * lines_per_col
                prompt_columns.append(all_lines[start:start + lines_per_col])
            # Height = tallest column
            line_bbox = tmp_d.textbbox((0, 0), "Ay", font=prompt_font)
            prompt_line_h = int((line_bbox[3] - line_bbox[1]) * 1.3)
            max_col_lines = max(
                (len(col) for col in prompt_columns), default=0)
            prompt_footer_h = prompt_line_h * max_col_lines + prompt_pad * 2

        sheet_h = grid_h + prompt_footer_h

        # Create sheet
        sheet = Image.new("RGB", (sheet_w, sheet_h), (24, 24, 28))
        draw = ImageDraw.Draw(sheet)

        # Draw title bar
        if title_val:
            t_font = _get_font(title_font_size)
            draw.rectangle([(0, 0), (sheet_w, title_h)], fill=(36, 36, 42))
            draw.text((2, 0), title_val, fill=(255, 220, 130), font=t_font)

        # Draw images + labels — flush side by side, per-row height
        for r, row_items in enumerate(rows_layout):
            x_cursor = 0
            cy = row_y_offsets[r]
            rh = row_heights[r]

            for pil_img, label in row_items:
                img_w, img_h = pil_img.size

                # Vertically center within row if heights differ
                off_y = (rh - img_h) // 2
                sheet.paste(pil_img, (x_cursor, cy + off_y))

                # Thin border
                draw.rectangle(
                    [(x_cursor - 1, cy + off_y - 1),
                     (x_cursor + img_w, cy + off_y + img_h)],
                    outline=(60, 60, 66), width=1
                )

                # Label bar under this row's image area
                bar_y0 = cy + rh
                bar_y1 = bar_y0 + label_h
                draw.rectangle(
                    [(x_cursor, bar_y0), (x_cursor + img_w, bar_y1)],
                    fill=(32, 32, 38))

                # Per-image font: proportional to THIS image's width
                img_font_size = min(max(img_w // 20, 14), 120)
                img_font = _get_font(img_font_size)

                # Center text in label bar
                text_bbox = draw.textbbox((0, 0), label, font=img_font)
                text_w = text_bbox[2] - text_bbox[0]
                text_h = text_bbox[3] - text_bbox[1]
                label_x = x_cursor + (img_w - text_w) // 2
                label_y = bar_y0 + (label_h - text_h) // 2

                if label.startswith("#"):
                    fill_color = (130, 200, 255)
                else:
                    fill_color = (235, 235, 230)
                draw.text((label_x, label_y), label, fill=fill_color, font=img_font)

                x_cursor += img_w + gap

        # -- Prompt footer: multi-column layout ---
        if has_prompt and prompt_font and prompt_columns:
            fy = grid_h
            # Background
            draw.rectangle([(0, fy), (sheet_w, sheet_h)], fill=(26, 26, 32))
            # Separator line
            draw.line([(0, fy), (sheet_w, fy)], fill=(55, 55, 65), width=2)
            # Draw each text column
            for c, col_lines in enumerate(prompt_columns):
                col_x = prompt_pad + c * (prompt_col_w + prompt_col_gap)
                # Vertical divider between columns
                if c > 0:
                    div_x = col_x - prompt_col_gap // 2
                    draw.line(
                        [(div_x, fy + prompt_pad // 2),
                         (div_x, sheet_h - prompt_pad // 2)],
                        fill=(50, 50, 58), width=1)
                # Draw lines — left-aligned
                for i, line in enumerate(col_lines):
                    ly = fy + prompt_pad + i * prompt_line_h
                    draw.text((col_x, ly), line,
                              fill=(205, 205, 200), font=prompt_font)

        # -- Watermark "happyin" — top-right, over images ---
        wm_font = _get_font(max(24, max_font // 2))
        wm_text = "happyin"
        wm_bbox = draw.textbbox((0, 0), wm_text, font=wm_font)
        wm_w = wm_bbox[2] - wm_bbox[0]
        wm_h = wm_bbox[3] - wm_bbox[1]
        wm_x = sheet_w - wm_w - 10
        wm_y = title_h + 6
        draw.text((wm_x, wm_y), wm_text, fill=(130, 130, 140), font=wm_font)

        # Convert back to tensor
        sheet_np = np.array(sheet).astype(np.float32) / 255.0
        sheet_tensor = torch.from_numpy(sheet_np).unsqueeze(0)

        print(f"[HappyinContactSheet] {n} items, {nrows} rows, "
              f"{sheet_w}x{sheet_h}px"
              + (f" +prompt" if has_prompt else ""))
        return (sheet_tensor,)
