"""Fetch demo data from the internet.

  python scripts/get_sample_video.py --test-set
      Public face-recognition test set (ageitgey/face_recognition examples): a few
      portraits + a short clip. The portraits are placed in data/faces/T00x/ and the
      people are added to students.csv as batch DEMO.

  python scripts/get_sample_video.py --url "https://www.youtube.com/watch?v=..."
  python scripts/get_sample_video.py --search "students sitting in classroom lecture"
      Classroom footage via yt-dlp, saved as data/samples/classroom.mp4.
"""
import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
SAMPLES = ROOT / "data" / "samples"
RAW = "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/"
TEST_FILES = ["obama.jpg", "obama2.jpg", "biden.jpg", "two_people.jpg", "lin-manuel-miranda.png",
              "alex-lacamoire.png", "hamilton_clip.mp4", "short_hamilton_clip.mp4"]
TEST_PEOPLE = [("T001", "Barack Obama", ["obama.jpg", "obama2.jpg"]),
               ("T002", "Joe Biden", ["biden.jpg"]),
               ("T003", "Lin-Manuel Miranda", ["lin-manuel-miranda.png"]),
               ("T004", "Alex Lacamoire", ["alex-lacamoire.png"])]


def fetch(url, dest):
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  have   {dest.name}")
        return
    print(f"  fetch  {dest.name}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def test_set():
    import pandas as pd
    out = SAMPLES / "test_set"
    out.mkdir(parents=True, exist_ok=True)
    for name in TEST_FILES:
        fetch(RAW + name, out / name)
    csv = ROOT / "data" / "students.csv"
    df = pd.read_csv(csv, dtype=str) if csv.exists() else pd.DataFrame(columns=["roll_no", "name", "batch"])
    for roll, name, files in TEST_PEOPLE:
        folder = ROOT / "data" / "faces" / roll
        folder.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy(out / f, folder / f)
        df = df[df["roll_no"] != roll]
        df.loc[len(df)] = [roll, name, "DEMO"]
    df.sort_values("roll_no").to_csv(csv, index=False)
    print(f"Test set ready in {out}\nNext: python scripts/enroll.py")


def classroom(target, height, seconds, name):
    import yt_dlp
    SAMPLES.mkdir(parents=True, exist_ok=True)
    opts = {
        # video-only H.264 stream: no audio needed, so no ffmpeg merge is required
        "format": f"bv[ext=mp4][vcodec^=avc1][height<={height}]/b[ext=mp4][height<={height}]/b",
        "outtmpl": str(SAMPLES / f"{name}.%(ext)s"),
        "noplaylist": True, "overwrites": True, "quiet": False,
    }
    try:
        import imageio_ffmpeg
        # yt-dlp only recognises a binary called ffmpeg(.exe), so keep a copy under that name
        tools = ROOT / "tools"
        tools.mkdir(exist_ok=True)
        exe = tools / ("ffmpeg.exe" if sys.platform == "win32" else "ffmpeg")
        if not exe.exists():
            shutil.copy(imageio_ffmpeg.get_ffmpeg_exe(), exe)
        opts["ffmpeg_location"] = str(tools)
        # the downloader checks this context variable, which only the yt-dlp CLI sets by itself
        yt_dlp.postprocessor.FFmpegPostProcessor._ffmpeg_location.set(str(tools))
        if seconds:
            opts["download_ranges"] = yt_dlp.utils.download_range_func(None, [(0, seconds)])
            opts["force_keyframes_at_cuts"] = False
    except ImportError:
        if seconds:
            print("imageio-ffmpeg not installed: downloading the whole video instead of a section.")
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(target, download=True)
        if "entries" in info:
            info = info["entries"][0]
    print(f"\nSaved: {info.get('title')}  ({info.get('webpage_url')})")
    print(f"File : {SAMPLES / (name + '.' + info.get('ext', 'mp4'))}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-set", action="store_true", help="download the public face test set")
    ap.add_argument("--url", help="video URL for yt-dlp")
    ap.add_argument("--search", help="YouTube search text (first result is downloaded)")
    ap.add_argument("--height", type=int, default=1080, help="maximum video height")
    ap.add_argument("--seconds", type=int, default=0, help="only the first N seconds (0 = all)")
    ap.add_argument("--name", default="classroom", help="output file name without extension")
    a = ap.parse_args()
    if a.test_set:
        test_set()
    if a.url or a.search:
        classroom(a.url or f"ytsearch1:{a.search}", a.height, a.seconds, a.name)
    if not (a.test_set or a.url or a.search):
        ap.print_help()
