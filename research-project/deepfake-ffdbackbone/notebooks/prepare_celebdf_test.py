"""
คัดเฉพาะ 518 วิดีโอใน test set ของ Celeb-DF v2 (ตาม List_of_testing_videos.txt)
แล้วรวมเป็น zip ไฟล์เดียวสำหรับอัปเป็น private Kaggle Dataset

วิธีใช้ (Anaconda Prompt หรือ Terminal):
    cd "D:\\Master\\Master term 1\\MSC DPML 2026\\paper\\kaggle"
    python prepare_celebdf_test.py

ถ้าโฟลเดอร์ Celeb-DF-v2 อยู่ที่อื่น:
    python prepare_celebdf_test.py --src "path\\to\\Celeb-DF-v2"

ผลลัพธ์: celebdf_v2_test518.zip (อยู่โฟลเดอร์เดียวกับสคริปต์)
    celebdf_v2_test518/
        List_of_testing_videos.txt
        manifest.csv                (path, label_official, label_fake, size, md5)
        Celeb-real/*.mp4            108 คลิป
        YouTube-real/*.mp4           70 คลิป
        Celeb-synthesis/*.mp4       340 คลิป
ใช้แค่ standard library ของ Python ไม่ต้องติดตั้งอะไรเพิ่ม
"""
import argparse
import csv
import hashlib
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED = {"Celeb-real": 108, "YouTube-real": 70, "Celeb-synthesis": 340}


def md5sum(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(HERE.parent / "Celeb-DF-v2"), help="โฟลเดอร์ Celeb-DF-v2 ที่แตก zip แล้ว")
    ap.add_argument("--out", default=str(HERE / "celebdf_v2_test518.zip"))
    args = ap.parse_args()

    src = Path(args.src)
    list_file = src / "List_of_testing_videos.txt"
    if not list_file.exists():
        sys.exit(f"[ERROR] ไม่พบ {list_file}")

    rows = []
    for line in list_file.read_text(encoding="utf-8").split("\n"):
        parts = line.split()
        if len(parts) == 2:
            rows.append((int(parts[0]), parts[1].replace("\\", "/")))
    print(f"รายการใน List_of_testing_videos.txt: {len(rows)} คลิป")

    missing, empty, counts = [], [], {}
    for label, rel in rows:
        p = src / rel
        if not p.exists():
            missing.append(rel)
        elif p.stat().st_size == 0:
            empty.append(rel)
        counts[rel.split("/")[0]] = counts.get(rel.split("/")[0], 0) + 1
    print("จำนวนแยกตามโฟลเดอร์:", counts)
    if counts != EXPECTED:
        print(f"[WARN] จำนวนไม่ตรงกับที่คาด {EXPECTED}")
    if missing or empty:
        print(f"[ERROR] ไม่พบไฟล์ {len(missing)} คลิป, ไฟล์ว่าง {len(empty)} คลิป เช่น:", (missing + empty)[:10])
        print("ตรวจว่าแตก Celeb-DF-v2.zip ครบ (แนะนำ 7-Zip) แล้วรันใหม่")
        sys.exit(1)

    out = Path(args.out)
    tmp = out.with_suffix(".zip.part")
    root = "celebdf_v2_test518"
    manifest = [("path", "label_official(1=real)", "label_fake(1=fake)", "size_bytes", "md5")]
    total = 0
    # mp4 บีบอัดแล้ว -> ใช้ ZIP_STORED เร็วกว่าและขนาดเท่าเดิม
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        z.write(list_file, f"{root}/List_of_testing_videos.txt")
        for i, (label, rel) in enumerate(rows, 1):
            p = src / rel
            size = p.stat().st_size
            manifest.append((rel, label, 1 - label, size, md5sum(p)))
            z.write(p, f"{root}/{rel}")
            total += size
            if i % 50 == 0 or i == len(rows):
                print(f"  {i}/{len(rows)} คลิป ({total/1e6:.0f} MB)")
        man_path = HERE / "_manifest_tmp.csv"
        with open(man_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(manifest)
        z.write(man_path, f"{root}/manifest.csv")
        man_path.unlink()
    tmp.replace(out)

    with zipfile.ZipFile(out) as z:
        bad = z.testzip()
    print(f"\nเสร็จ: {out}")
    print(f"ขนาด {out.stat().st_size/1e6:.0f} MB, ตรวจ zip: {'OK' if bad is None else 'เสียที่ ' + bad}")
    print("ขั้นต่อไป: kaggle.com -> Datasets -> New Dataset -> อัปไฟล์ zip นี้ -> ตั้งเป็น Private")


if __name__ == "__main__":
    main()
