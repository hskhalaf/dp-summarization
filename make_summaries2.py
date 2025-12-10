import argparse
import requests
import time
import json

from dpsummarizer import DPSummarizer, FrozenLLM
from dpsummarizer.log import set_level, logging
from dpsummarizer.utils import save_adapter, load_adapter
from dpsummarizer.io import IO


def main(args):
    start = time.time()
    try:
        llm = FrozenLLM(model_name=args.model_name)
        summarizer = DPSummarizer(args.seed)
        io = IO()
        
        public_dataset  = io.read_documents("summary_data/train/", max_docs=args.max_public_docs)
        private_dataset = io.read_documents("summary_data/test/", max_docs=args.max_private_docs)

        if not private_dataset:
            logging.error("No private documents loaded; nothing to summarize.")
            return

        instruction_template = (
            "You are summarizing customer reviews for the product:\n"
            "Title: {}\n"
            "Categories: {}\n\n"
            "Write exactly two sentences of the form:\n"
            "\"The product is a [TYPE]. Customers praise its [ASPECT_1] "
            "and [ASPECT_2], but some complain about [ISSUE_1] and [ISSUE_2].\"\n"
        )
        
        if args.load_adapter:
            adapter = summarizer.train(
                llm=llm,
                public_dataset=public_dataset,
                instruction_template=instruction_template,
                m=args.m,
                lr=args.lr,
                num_epochs=0,  # Skip training, just initialize
                max_reviews_per_product=args.max_reviews_per_doc,
                weight_decay=args.weight_decay,
                batch_size=args.batch_size,
                docs_per_epoch=args.docs_per_epoch,
            )
            if not load_adapter(adapter, args.load_adapter):
                logging.error("Failed to load adapter. Exiting.")
                return
        else:
            adapter = summarizer.train(
                llm=llm,
                public_dataset=public_dataset,
                instruction_template=instruction_template,
                m=args.m,
                lr=args.lr,
                num_epochs=args.num_epochs,
                max_reviews_per_product=args.max_reviews_per_doc,
                weight_decay=args.weight_decay,
                batch_size=args.batch_size,
                docs_per_epoch=args.docs_per_epoch,
            )
            if args.save_adapter:
                save_adapter(adapter, args.model_name)
        
        if args.no_dp:
            args.eps = [-1] # value gets ignored anyways

        products_output: list[dict] = []

        for idx, (reviews, reference_summary, metadata) in enumerate(private_dataset):
            logging.info(f"Summarizing product {idx + 1}/{len(private_dataset)} (k={args.k})")
            product_summaries: list[dict] = []

            for eps in args.eps:
                if not args.no_dp:
                    logging.info(f"Evaluating with epsilon={eps}...")
                summary = summarizer.summarize(
                    llm=llm,
                    adapter=adapter,
                    private_reviews=reviews,
                    metadata=metadata,
                    instruction_template=instruction_template,
                    max_new_tokens=50,
                    C=20.0,
                    epsilon=eps,
                    delta=args.delta,
                    no_dp=args.no_dp,
                    k=args.k,
                    batch_size=args.batch_size,
                )

                if args.output in ("console", "all"):
                    print(summary)

                product_summaries.append({"epsilon": eps, "summary": summary})

            products_output.append({
                "product_metadata": metadata,
                "reference_summary": reference_summary,
                "summaries": product_summaries,
            })

        if args.output in ("json", "all"):
            io.export_multi_summaries(
                args=vars(args),
                products=products_output,
                output_path=args.output_file,
            )
    except Exception as e:
        logging.error(f"An error occurred during summarization: {e}", exc_info=True)
    finally:
        # Send notification if topic provided
        if args.notify:
            send_ntfy_notification(time.time() - start)


def send_ntfy_notification(time):
    try:
        title = f"Completed in {time / 60:.1f} min"
        message = f"""Check terminal for results."""
        
        url = f"https://ntfy.sh/dp-summarization"
        
        logging.info(f"Sending notification...")
        
        response = requests.post(
            url,
            data=message.encode('utf-8'),
            headers={
                "Title": title,
                "Priority": "default",
                "Tags": "white_check_mark,robot"
            }
        )
        
        if response.status_code == 200:
            logging.info(f"Notification sent to ntfy.sh/dp-summarization")
        else:
            logging.warning(f"Failed to send notification: {response.status_code}")
        
    except Exception as e:
        logging.error(f"Failed to send ntfy notification: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DP summarization.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "debug", "info", "warning", "error"],
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    parser.add_argument(
        "--output",
        type=str,
        choices=["console", "json", "all"],
        default="console",
        help="Output format: console, json, or all.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="results/summaries_export.json",
        help="Path for JSON output when using --output json/all (default: results/summaries_export.json)",
    )

    parser.add_argument(
        "--max-public-docs",
        type=int,
        default=None,
        help="Maximum number of public documents to use for training.",
    )
    parser.add_argument(
        "--max-private-docs",
        type=int,
        default=100,
        help="Maximum number of private (test) products to summarize.",
    )
    parser.add_argument(
        "--max-reviews-per-doc",
        type=int,
        default=100,
        help="Maximum number of reviews per private document.",
    )
    parser.add_argument(
        "--docs-per-epoch",
        type=int,
        default=None,
        help="Number of documents to sample per epoch. If None, uses all public docs. E.g., with 300 docs and --docs-per-epoch 10, each epoch trains on 10 randomly sampled docs.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Name of the pre-trained LLM model to use.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=5,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate for training.",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=32,
        help="Number of soft prompt tokens.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=5e-2,
        help="Weight decay for optimizer.",
    )
    parser.add_argument(
        "--save-adapter",
        type=bool,
        default=True,
        help="Save trained adapter with auto-versioning (e.g., Llama_3_2_1B_Instruct_v1.pt)",
    )
    parser.add_argument(
        "--load-adapter",
        type=str,
        default=None,
        help="Load pre-trained adapter from dpsummarizer/adapter_checkpoints/<name>.pt instead of training.",
    )

    dp_parser = parser.add_mutually_exclusive_group()
    dp_parser.add_argument(
        "--eps",
        type=lambda s: [float(item) for item in s.split(',')],
        default=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 120.0],
        help="Comma-separated epsilon values (e.g., '0.1,0.5,1.0').",
    )
    dp_parser.add_argument(
        "--no-dp",
        action="store_true",
        help="Disable differential privacy. Ignores --eps and --delta values.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-6,
        help="Delta value. Ignored if --no-dp is set.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="Number of soft prompts to generate. Uses basic composition if k>1.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of reviews to process in parallel during encoding.",
    )
    parser.add_argument(
        "--notify",
        action="store_true",
        default=False,
        help="Send notification when done. Check ntfy.sh/dp-summarization.",
    )
    args = parser.parse_args()

    assert args.delta >= 0.0, "Delta must be non-negative."
    assert all(eps > 0.0 for eps in args.eps), "All epsilon values must be positive."
    assert args.num_epochs >= 0, "Number of epochs must be non-negative."
    assert args.max_reviews_per_doc > 0, "Max reviews per document must be positive."
    assert args.k > 0, "k must be positive."
    assert args.batch_size > 0, "Batch size must be positive."
    assert args.docs_per_epoch is None or args.docs_per_epoch > 0, "Docs per epoch must be positive if specified."
    assert args.max_public_docs is None or args.max_public_docs > 0, "Max public docs must be positive if specified."
    assert args.max_private_docs > 0, "Max private docs must be positive."

    set_level(getattr(logging, args.log_level.upper()))
    main(args)
