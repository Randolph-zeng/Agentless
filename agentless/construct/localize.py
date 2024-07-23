import argparse
import concurrent.futures
import json
import os

from datasets import load_dataset
from tqdm import tqdm

from agentless.construct.FL import LLMFL
from agentless.util.preprocess_data import (
    filter_none_python,
    filter_out_test_files
)
from agentless.util.utils import (
    load_existing_instance_ids,
    setup_logger,
)
from get_repo_structure.get_repo_structure import (
    get_project_structure_from_scratch,
)

# SET THIS IF YOU WANT TO USE THE PREPROCESSED FILES
PROJECT_FILE_LOC = os.environ.get("PROJECT_FILE_LOC", None)


def localize_instance(
    bench_data, args, existing_instance_ids
):
    instance_id = bench_data["instance_id"]
    log_file = os.path.join(
        args.output_folder, "localization_logs", f"{instance_id}.log"
    )

    logger = setup_logger(log_file)
    logger.info(f"Processing bug {instance_id}")

    if instance_id in existing_instance_ids:
        logger.info(f"Skipping existing instance_id: {instance_id}")
        return

    # we need to get the project structure directly
    # ZZ: TODO add cache logic here 
    d = get_project_structure_from_scratch(
        bench_data["repo"], bench_data["base_commit"], instance_id, "/dataset/zszeng/AgentlessOutputs/playground"
    )

    logger.info(f"================ localize {instance_id} ================")

    problem_statement = bench_data["problem_statement"]
    structure = d["structure"]

    filter_none_python(structure)  # some basic filtering steps

    # filter out test files (unless its pytest)
    if not d["instance_id"].startswith("pytest"):
        filter_out_test_files(structure)

    found_files = []
    found_related_locs = []
    found_edit_locs = []
    additional_artifact_loc_file = None
    additional_artifact_loc_related = None
    additional_artifact_loc_edit_location = None
    file_traj, related_loc_traj, edit_loc_traj = {}, {}, {}

    # file level localization
    if args.file_level:
        fl = LLMFL(
            d["instance_id"],
            structure,
            problem_statement,
            args.model,
            args.backend,
            logger,
            args.match_partial_paths,
        )
        found_files, additional_artifact_loc_file, file_traj = fl.localize(
            mock=args.mock
        )

    # related class, functions, global var localization
    if args.related_level:
        if len(found_files) != 0:
            pred_files = found_files[: args.top_n]
            fl = LLMFL(
                d["instance_id"],
                structure,
                problem_statement,
                args.model,
                args.backend,
                logger,
                args.match_partial_paths,
            )
            additional_artifact_loc_related = []
            found_related_locs = []
            related_loc_traj = {}
            (
                found_related_locs,
                additional_artifact_loc_related,
                related_loc_traj,
            ) = fl.localize_function_from_compressed_files(
                pred_files, mock=args.mock
            )
            additional_artifact_loc_related = [additional_artifact_loc_related]
            

    if args.fine_grain_line_level:
        # Only supports the following args for now
        pred_files = found_files[: args.top_n]
        fl = LLMFL(
            instance_id,
            structure,
            problem_statement,
            args.model,
            args.backend,
            logger,
            args.match_partial_paths,
        )
        coarse_found_locs = {}
        for i, pred_file in enumerate(pred_files):
            if len(found_related_locs) > i:
                coarse_found_locs[pred_file] = found_related_locs[i]
        (
            found_edit_locs,
            additional_artifact_loc_edit_location,
            edit_loc_traj,
        ) = fl.localize_line_from_coarse_function_locs(
            pred_files,
            coarse_found_locs,
            context_window=args.context_window,
            add_space=args.add_space,
            no_line_number=args.no_line_number,
            sticky_scroll=args.sticky_scroll,
            mock=args.mock,
            temperature=args.temperature,
            num_samples=args.num_samples,
        )
        additional_artifact_loc_edit_location = [additional_artifact_loc_edit_location]

    with open(args.output_file, "a") as f:
        f.write(
            json.dumps(
                {
                    "instance_id": d["instance_id"],
                    "found_files": found_files,
                    "additional_artifact_loc_file": additional_artifact_loc_file,
                    "file_traj": file_traj,
                    "found_related_locs": found_related_locs,
                    "additional_artifact_loc_related": additional_artifact_loc_related,
                    "related_loc_traj": related_loc_traj,
                    "found_edit_locs": found_edit_locs,
                    "additional_artifact_loc_edit_location": additional_artifact_loc_edit_location,
                    "edit_loc_traj": edit_loc_traj,
                }
            )
            + "\n"
        )


def localize(args):
    swe_bench_data = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    existing_instance_ids = (
        load_existing_instance_ids(args.output_file) if args.skip_existing else set()
    )

    if args.num_threads == 1:
        for bug in swe_bench_data:
            localize_instance(
                bug, args, existing_instance_ids
            )
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.num_threads
        ) as executor:
            futures = [
                executor.submit(
                    localize_instance,
                    bug,
                    args,
                    existing_instance_ids,
                )
                for bug in swe_bench_data
            ]
            concurrent.futures.wait(futures)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="loc_outputs.jsonl")
    parser.add_argument("--top_n", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--add_space", action="store_true")
    parser.add_argument("--no_line_number", action="store_true")
    parser.add_argument("--sticky_scroll", action="store_true")
    parser.add_argument(
        "--match_partial_paths",
        action="store_true",
        help="Whether to match model generated files based on subdirectories of original repository if no full matches can be found",
    )
    parser.add_argument("--context_window", type=int, default=10)
    parser.add_argument(
        "--num_threads",
        type=int,
        default=1,
        help="Number of threads to use for creating API requests",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip localization of instance id's which already contain a localization in the output file.",
    )
    parser.add_argument(
        "--mock", action="store_true", help="Mock run to compute prompt tokens."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-2024-05-13",
        choices=["gpt-4o-2024-05-13", "deepseek-coder", "gpt-4o-mini-2024-07-18"],
    )
    parser.add_argument(
        "--backend", type=str, default="openai", choices=["openai", "deepseek"]
    )

    args = parser.parse_args()

    import os

    args.output_file = os.path.join(args.output_folder, args.output_file)

    assert (
        not os.path.exists(args.output_file) or args.skip_existing
    ), "Output file already exists and not set to skip existing localizations"

    
    assert (not "deepseek" in args.model) or (
        args.backend == "deepseek"
    ), "Must specify `--backend deepseek` if using a DeepSeek model"

    os.makedirs(os.path.join(args.output_folder, "localization_logs"), exist_ok=True)
    os.makedirs(args.output_folder, exist_ok=True)

    # write the arguments
    with open(f"{args.output_folder}/args.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    localize(args)


if __name__ == "__main__":
    main()
