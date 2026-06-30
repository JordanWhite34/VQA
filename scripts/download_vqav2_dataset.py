import json
import random
import zipfile
import urllib.request
from pathlib import Path
from tqdm import tqdm


ROOT = Path("data/vqav2_small")
RAW = ROOT / "raw"
IMG_ROOT = ROOT / "images"

RAW.mkdir(parents=True, exist_ok=True)
IMG_ROOT.mkdir(parents=True, exist_ok=True)

FILES = {
    "train_annotations": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip",
    "val_annotations": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
    "train_questions": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip",
    "val_questions": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
}


def download_file(url: str, out_path: Path):
    if out_path.exists():
        print(f"Exists: {out_path}")
        return

    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response:
        total = int(response.headers.get("Content-Length", 0))
        with open(out_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, unit_divisor=1024
        ) as bar:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                bar.update(len(chunk))


def unzip_file(zip_path: Path, out_dir: Path):
    marker = out_dir / f".unzipped_{zip_path.stem}"
    if marker.exists():
        return

    print(f"Unzipping {zip_path.name}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)

    marker.touch()


def coco_filename(split: str, image_id: int) -> str:
    return f"COCO_{split}_{image_id:012d}.jpg"


def coco_url(split: str, image_id: int) -> str:
    fname = coco_filename(split, image_id)
    return f"http://images.cocodataset.org/{split}/{fname}"


def download_coco_image(split: str, image_id: int):
    split_dir = IMG_ROOT / split
    split_dir.mkdir(parents=True, exist_ok=True)

    fname = coco_filename(split, image_id)
    out_path = split_dir / fname

    if out_path.exists():
        return str(out_path)

    url = coco_url(split, image_id)

    try:
        urllib.request.urlretrieve(url, out_path)
        return str(out_path)
    except Exception as e:
        print(f"Failed image {image_id}: {e}")
        return None


def make_subset(split: str, n: int, seed: int = 42):
    random.seed(seed)

    q_path = RAW / f"v2_OpenEnded_mscoco_{split}_questions.json"
    a_path = RAW / f"v2_mscoco_{split}_annotations.json"

    with open(q_path, "r", encoding="utf-8") as f:
        questions = json.load(f)["questions"]

    with open(a_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)["annotations"]

    ann_by_qid = {a["question_id"]: a for a in annotations}

    sampled_questions = random.sample(questions, min(n, len(questions)))

    rows = []
    image_ids = set()

    for q in sampled_questions:
        qid = q["question_id"]
        ann = ann_by_qid[qid]

        row = {
            "question_id": qid,
            "image_id": q["image_id"],
            "image_file": coco_filename(split, q["image_id"]),
            "image_path": str(IMG_ROOT / split / coco_filename(split, q["image_id"])),
            "question": q["question"],
            "answer": ann["multiple_choice_answer"],
            "answers": [x["answer"] for x in ann["answers"]],
            "question_type": ann["question_type"],
            "answer_type": ann["answer_type"],
            "split": split,
        }

        rows.append(row)
        image_ids.add(q["image_id"])

    print(f"{split}: {len(rows)} questions, {len(image_ids)} unique images")

    for image_id in tqdm(sorted(image_ids), desc=f"Downloading {split} images"):
        download_coco_image(split, image_id)

    out_path = ROOT / f"{split}_subset.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {out_path}")


def main():
    for name, url in FILES.items():
        zip_path = RAW / f"{name}.zip"
        download_file(url, zip_path)
        unzip_file(zip_path, RAW)

    make_subset("train2014", n=5000)
    make_subset("val2014", n=1000)


if __name__ == "__main__":
    main()