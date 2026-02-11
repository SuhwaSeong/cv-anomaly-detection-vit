import json
import base64
import re
from pathlib import Path
from llm import label
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import csv


class GenerateLabel(label):
    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        context_prompt_path: Path,
        llm_deployed_model: str = "cv-anomaly-gpt4o-mini",
    ):
        super().__init__(context_prompt_path=context_prompt_path, llm_deployed_model=llm_deployed_model)
        self.input_dir = input_dir
        self.output_dir = output_dir

    def encode_image(self, image_path: Path) -> str:
        return base64.b64encode(image_path.read_bytes()).decode("utf-8")

    def parse_json_str(self, json_str: str) -> dict:
        match = re.search(r"```json\s*(.*?)\s*```", json_str, re.DOTALL)
        json_data = match.group(1).strip() if match else json_str.strip()
        return json.loads(json_data)

    def process_one(self, image_path: Path) -> tuple[str, dict] | tuple[str, None]:
        """
        Returns:
            (stem, response_dict) on success
            (stem, None) on failure
        """
        try:
            base64_image = self.encode_image(image_path)
            json_response = self.run(base64_image)
            response = self.parse_json_str(json_response)
            return image_path.stem, response
        except Exception as e:
            print(f"Failed: {image_path.name} -> {e}")
            return image_path.stem, None

    @staticmethod
    def reset_outputs(output_dir: Path):
        output_dir.mkdir(exist_ok=True)

        # Delete old merged outputs
        for p in output_dir.glob("labels_*.json"):
            p.unlink()
        for p in [output_dir / "labels.json", output_dir / "labels.csv"]:
            if p.exists():
                p.unlink()

    @staticmethod
    def to_df(labels_dict: dict) -> pd.DataFrame:
        df = pd.DataFrame.from_dict(labels_dict, orient="index").reset_index()
        df.rename(columns={"index": "image_name"}, inplace=True)

        # Normalize key
        if "animal" in df.columns:
            df.rename(columns={"animal": "label"}, inplace=True)

        required_cols = ["image_name", "label", "notes"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in label data: {missing}")

        return df[required_cols]

    def run_all(self, max_workers: int = 1) -> tuple[dict, list[str]]:
        files = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            files += list(self.input_dir.glob(ext))
        files = sorted(files)

        if not files:
            raise FileNotFoundError(f"No images found in: {self.input_dir}")

        labels = {}
        failed = []

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(self.process_one, f): f for f in files}
            for fut in tqdm(as_completed(futures), total=len(files), desc="Labeling..."):
                stem, resp = fut.result()
                if resp is None:
                    failed.append(stem)
                else:
                    labels[stem] = resp

        return labels, failed


if __name__ == "__main__":
    context_prompt = Path("prompts") / "label.md"
    input_dir = Path("frames")
    output_dir = Path("labels")

    gen = GenerateLabel(
        input_dir=input_dir,
        output_dir=output_dir,
        context_prompt_path=context_prompt,
    )

    # 1) Hard reset outputs (delete old JSON/CSV)
    gen.reset_outputs(output_dir)
    print("Reset done: old labels.json / labels.csv / labels_*.json deleted.")

    # 2) Label all frames (keep max_workers low to reduce 429)
    labels_dict, failed = gen.run_all(max_workers=1)

    # 3) Save merged JSON
    output_dir.mkdir(exist_ok=True)
    labels_json_path = output_dir / "labels.json"
    with labels_json_path.open("w", encoding="utf-8") as f:
        json.dump(labels_dict, f, ensure_ascii=False, indent=4)

    # 4) Save merged CSV
    df = gen.to_df(labels_dict)
    labels_csv_path = output_dir / "labels.csv"
    df.to_csv(labels_csv_path, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)

    print(f"Saved JSON: {labels_json_path}")
    print(f"Saved CSV : {labels_csv_path}")

    if failed:
        print(f"Failed frames ({len(failed)}): {failed}")
