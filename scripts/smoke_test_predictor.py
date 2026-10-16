"""
Smoke test CLI script for ConversationGuardianPredictor.

Accepts dialogue message strings and prints a formatted JSON prediction.
Can be executed with custom messages via --messages or run across all supported
checkpoints (n = 1, 3, 5) by default.
"""

import sys
import json
import argparse
from pathlib import Path

# Ensure src directory is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from conversation_guardian.ml.predictor import ConversationGuardianPredictor


def main():
    parser = argparse.ArgumentParser(description="Smoke test for ConversationGuardianPredictor")
    parser.add_argument(
        "--messages",
        nargs="+",
        type=str,
        default=None,
        help="List of 1, 3, or 5 observed message strings for prediction",
    )
    parser.add_argument(
        "--bundles-dir",
        type=str,
        default=str(ROOT_DIR / "models" / "selected_predictors"),
        help="Path to directory containing deployable predictor bundles",
    )

    args = parser.parse_args()
    bundles_path = Path(args.bundles_dir)

    print("=" * 80)
    print("CONVERSATIONGUARDIAN — TASK 4: PREDICTION SERVICE SMOKE TEST")
    print(f"Bundles directory: {bundles_path}")
    print("=" * 80)

    predictor = ConversationGuardianPredictor(bundles_dir=bundles_path)

    if args.messages is not None:
        # Run prediction on user-supplied messages
        print(f"\n[Custom Inference] Running prediction on {len(args.messages)} supplied message(s)...")
        result = predictor.predict(args.messages)
        print(json.dumps(result, indent=2))
    else:
        # Run default smoke test across all three supported checkpoints
        test_cases = {
            1: [
                "Thank you for checking the diff, much appreciated."
            ],
            3: [
                "Hello, I wanted to discuss the recent changes to the lead section.",
                "I disagree with your removal of the cited source.",
                "Stop being an idiot and listen to what the talk page consensus says!",
            ],
            5: [
                "First edit looked fine to me.",
                "I see your point regarding the secondary reference.",
                "Let us move this to the talk page before making further edits.",
                "Please stop reverting while discussion is ongoing.",
                "You are an arrogant clown, stop lying about the references!",
            ],
        }

        for n, msgs in test_cases.items():
            print(f"\n" + "-" * 80)
            print(f"CHECKPOINT n = {n} OBSERVED MESSAGE(S)")
            print("-" * 80)
            result = predictor.predict(msgs)
            print(json.dumps(result, indent=2))

    print("\n" + "=" * 80)
    print("SMOKE TEST COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
