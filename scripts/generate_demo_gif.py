from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

out_dir = Path("docs/images")
out_dir.mkdir(parents=True, exist_ok=True)

TARGET_WIDTH = 1000
TARGET_HEIGHT = 625
BG_COLOR = (11, 15, 25)  # #0b0f19 dark obsidian

def prepare_frame(image_path: Path, title: str) -> Image.Image:
    base = Image.new("RGB", (TARGET_WIDTH, TARGET_HEIGHT), BG_COLOR)
    
    # Load and resize source image preserving aspect ratio
    src = Image.open(image_path).convert("RGBA")
    
    # Calculate fitting
    max_w = TARGET_WIDTH - 40
    max_h = TARGET_HEIGHT - 60
    
    src_ratio = src.width / src.height
    box_ratio = max_w / max_h
    
    if src_ratio > box_ratio:
        new_w = max_w
        new_h = int(new_w / src_ratio)
    else:
        new_h = max_h
        new_w = int(new_h * src_ratio)
        
    src_resized = src.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Center image in container below header
    offset_x = (TARGET_WIDTH - new_w) // 2
    offset_y = 50 + (max_h - new_h) // 2
    
    base.paste(src_resized, (offset_x, offset_y), src_resized)
    
    # Draw polished top banner
    draw = ImageDraw.Draw(base)
    draw.rectangle([(0, 0), (TARGET_WIDTH, 42)], fill=(15, 23, 42))  # #0f172a
    draw.line([(0, 42), (TARGET_WIDTH, 42)], fill=(30, 41, 59), width=2)
    
    # macOS style traffic light dots
    draw.ellipse([(16, 15), (26, 25)], fill=(239, 68, 68))
    draw.ellipse([(34, 15), (44, 25)], fill=(245, 158, 11))
    draw.ellipse([(52, 15), (62, 25)], fill=(16, 185, 129))
    
    # Title badge text
    draw.text((80, 13), f"LoRAForge Control Plane // {title}", fill=(148, 163, 184))
    draw.text((TARGET_WIDTH - 180, 13), "v0.1.0 • BARE-METAL", fill=(56, 189, 248))
    
    return base

# Prepare slides
slides_info = [
    (Path("docs/images/dashboard.png"), "Predictive VRAM Profile & Allocation Matrix", 3200),
    (Path("docs/images/cli_vram.png"), "CLI Analytical VRAM Profiler & Sharding", 2800),
    (Path("docs/images/cli_autotune.png"), "Hardware Auto-Tuner & Parameter Search", 2600),
    (Path("docs/images/dashboard_training.png"), "Live Training Supervisor & Watchdog Telemetry", 3200),
    (Path("docs/images/loraforge_adapter_merger.png"), "Multi-LoRA Adapter Fusion (TIES / DARE / SVD)", 2800),
]

frames = []
durations = []

for path, title, dur in slides_info:
    if path.is_file():
        frame = prepare_frame(path, title)
        frames.append(frame)
        durations.append(dur)
        print(f"Processed frame: {title} ({dur}ms)")

if frames:
    gif_path = out_dir / "loraforge_demo.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    # Also save as demo.gif
    frames[0].save(
        out_dir / "demo.gif",
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    print(f"[OK] Successfully generated {gif_path} ({gif_path.stat().st_size} bytes)")
