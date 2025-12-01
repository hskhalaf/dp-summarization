import argparse

from dpsummarizer import DPSummarizer, FrozenLLM
from dpsummarizer.log import set_level, logging

def main(args):
    llm = FrozenLLM(model_name="meta-llama/Llama-3.2-3B-Instruct")
    summarizer = DPSummarizer()

    public_dataset  = summarizer.read_documents("summary_data/train/", max_docs=args.max_public_docs)
    private_dataset = summarizer.read_documents("summary_data/test/", max_docs=1)

    # instruction_template = (
    #     "Write one concise summary following this pattern:\n"
    #     '"The product is a [TYPE]. Customers praise its [ASPECT_1] and [ASPECT_2], '
    #     'but some complain about [ISSUE_1] and [ISSUE_2]."\n'
    #     "Replace every bracketed token with specific phrases from the review. "
    #     "Do not invent details or leave placeholders. Do not say anything more "
    #     "than what has been asked of you. Only say the exact quote listed above. "
    #     "Do not generate example summaries, only do so based on the review given to you. \n"
    # )
    instruction_template = "Summary:"

    adapter = summarizer.train(
        llm=llm,
        public_dataset=public_dataset,
        instruction_template=instruction_template,
        m=32,
        lr=3e-4,
        num_epochs=10,
        max_reviews_per_product=args.max_reviews_per_doc,
    )

    if args.eps is not None:
        epsilons = [args.eps]
    else:
        epsilons = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]

    for eps in epsilons:
        logging.info(f"Evaluating with epsilon={eps}...")
        summary = summarizer.summarize(
            llm=llm,
            adapter=adapter,
            private_reviews=private_dataset[0][0],  # use reviews from the first example
            instruction_template=instruction_template,
            max_new_tokens=64,
            C=1.0,
            epsilon=eps,
            delta=args.delta,
        )

        print(f"Summary with ε={eps}:")
        print(summary)


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
        "--eps",
        type=float,
        default=None,
        help="Epsilon value. If not set, evaluates multiple epsilons.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        help="Delta value.",
    )
    parser.add_argument(
        "--max-public-docs",
        type=int,
        default=None,
        help="Maximum number of public documents to use for training.",
    )
    parser.add_argument(
        "--max-reviews-per-doc",
        type=int,
        default=100,
        help="Maximum number of reviews per private document.",
    )
    args = parser.parse_args()
    set_level(getattr(logging, args.log_level.upper()))
    main(args)
