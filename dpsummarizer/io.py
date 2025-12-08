from pathlib import Path
import json

import logging

class IO:
    def __init__(self):
        self.summaries: list[dict] = []

    def read_documents(self, file_path: str, max_docs: int | None = None) -> list[tuple[list[str], str, dict]]:
        """
        Read all .json files under the given path and return a list of
        (public_reviews, public_summary, metadata) tuples suitable for training.

        - public_reviews: list[str] of review texts (title + text).
        - public_summary: one concise reference summary (verdict if present, else compact pros/cons).
        - metadata: dict with price_bucket, rating.

        :param file_path: Directory containing .json files (scans recursively).
        :type file_path: str
        :param max_docs: Maximum number of documents to read. If None, read all.
        :type max_docs: int | None
        
        :return: List of (reviews, summary, metadata) tuples.
        """
        root = Path(file_path)
        if not root.exists():
            logging.warning(f"Path not found: {file_path}")
            return []

        logging.info(f"Reading documents from: {file_path}")

        pairs: list[tuple[list[str], str, dict]] = []
        count_files = 0
        count_used = 0

        for fp in root.rglob("*.json"):
            count_files += 1

            try:
                with fp.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                logging.warning(f"Failed to read {fp}: {e}")
                continue

            # build reviews
            reviews: list[str] = []
            for r in data.get("customer_reviews", []):
                title = r.get("title", "") or ""
                text = r.get("text", "") or ""
                txt = f"{title} {text}".strip()
                if txt:
                    reviews.append(txt)

            # Skip if no usable reviews
            if not reviews:
                continue

            # build summary
            summary: str | None = None
            ws = data.get("website_summaries", [])

            if isinstance(ws, list) and ws:
                first = ws[0] if isinstance(ws[0], dict) else {}
                verdict = first.get("verdict") if isinstance(first, dict) else None

                # Primary: use verdict if non-empty
                if isinstance(verdict, str) and verdict.strip():
                    summary = verdict.strip()
                else:
                    # Fallback: compact pros/cons across all entries
                    pros: list[str] = []
                    cons: list[str] = []
                    for s in ws:
                        if not isinstance(s, dict):
                            continue
                        if s.get("pros"):
                            pros.extend([str(p) for p in s.get("pros", []) if p])
                        if s.get("cons"):
                            cons.extend([str(c) for c in s.get("cons", []) if c])

                    if pros or cons:
                        pros_str = "; ".join(pros[:6]) if pros else "N/A"
                        cons_str = "; ".join(cons[:6]) if cons else "N/A"
                        summary = f"Pros: {pros_str}. Cons: {cons_str}."

            # Skip if we could not get any summary/label
            if not summary:
                continue

            # build metadata
            product_meta: dict = data.get("product_meta", {}) or {}

            price_str = product_meta.get("price", "")
            try:
                price = float(str(price_str).replace("$", "").replace(",", ""))
                price_bucket = 0 if price < 15 else (1 if price < 35 else 2)
            except Exception:
                price_bucket = 1  # Default to mid-range if unknown / unparsable

            try:
                rating = float(product_meta.get("rating", 3.0) or 3.0)
            except Exception:
                rating = 3.0

            title = product_meta.get("title", "")
            categories = product_meta.get("categories", [])

            metadata = {
                "price_bucket": price_bucket,
                "rating": rating,
                "title": title,
                "categories": categories,
                "source_path": str(fp),
            }

            # append to list
            pairs.append((reviews, summary, metadata))
            count_used += 1

            if max_docs is not None and count_used >= max_docs:
                logging.info(f"Reached max_docs={max_docs}; stopping early.")
                break

        logging.info(f"Scanned {count_files} JSON file(s); using {count_used} with reviews+summary.")
        return pairs
    
    def export_summaries(self, args: dict, product_metadata: dict, output_path: str | None = None) -> None:
        """
        Exports all summaries in self.summaries to a JSON file along with
        the provided command-line arguments and product metadata.
        
        :param args: All command-line arguments used to run the summarization.
        :type args: dict
        :param product_metadata: Metadata about the product being summarized.
        :type product_metadata: dict
        """
        output = {
            "args": args,
            "product_metadata": product_metadata,
            "summaries": self.summaries,
        }

        path = Path(output_path) if output_path else Path("results/summaries_export.json")
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with path.open("w", encoding="utf-8") as f:
                json.dump(output, f, indent=2)
            logging.info(f"Exported summaries to {path}")
        except Exception as e:
            logging.error(f"Failed to export summaries to {path}: {e}")

    