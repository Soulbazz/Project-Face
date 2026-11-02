import os
import subprocess

chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
html_path = os.path.abspath("reports/slide_results_blocks.html")
norm_url_base = html_path.replace("\\", "/")

for slide in range(1, 5):
    out_img = os.path.abspath(f"slide_{slide}.png")
    url = f"file:///{norm_url_base}?slide={slide}&capture=1"
    cmd = [
        chrome_path,
        "--headless=new",
        "--window-size=1920,1080",
        "--hide-scrollbars",
        "--virtual-time-budget=2000",
        f"--screenshot={out_img}",
        url,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if os.path.exists(out_img):
        print(f"Slide {slide} captured successfully: {os.path.getsize(out_img):,} bytes")
    else:
        print(f"Failed to capture slide {slide}: {res.stderr}")

