import concurrent.futures
import os, json, argparse
from difflib import unified_diff
from datasets import load_dataset
from agentless.construct.FL import LLMFL
from agentless.util.model import make_model
from agentless.util.utils import (load_existing_instance_ids, setup_logger)
from get_repo_structure.get_repo_structure import (get_project_structure_from_scratch)
from agentless.construct.prompts import (
    with_scope_explanation,
    repair_prompt_combine_topn,
    repair_prompt_combine_topn_cot,
    repair_relevant_file_instruction,
    repair_prompt_combine_topn_cot_diff,
    repair_relevant_file_with_scope_instruction,
    repair_relevant_file_with_suspicious_loc_instruction
)
from agentless.util.postprocess_data import (
    lint_code,
    check_syntax,
    fake_git_repo,
    remove_empty_lines,
    parse_edit_commands,
    extract_python_blocks,
    parse_diff_edit_commands,
    split_edit_multifile_commands,
    check_code_differ_by_just_empty_lines,
)
from agentless.util.preprocess_data import (
    line_wrap_content,
    filter_none_python,
    filter_none_python,
    transfer_arb_locs_to_locs,
    get_full_file_paths_and_classes_and_functions,
)


def localize_instance(bench_data, args, existing_instance_ids, logger):
    instance_id = bench_data["instance_id"]
    logger.info(f"Processing bug {instance_id}")

    if instance_id in existing_instance_ids:
        logger.info(f"Skipping existing instance_id: {instance_id}")
        return

    # we need to get the project structure directly
    # ZZ: TODO support multi-thread logic here  
    d = get_project_structure_from_scratch(
        bench_data["repo"], bench_data["base_commit"], instance_id, "/dataset/zszeng/AgentlessOutputs/playground"
    )

    logger.info(f"================ localize {instance_id} ================")

    problem_statement = bench_data["problem_statement"]
    structure = d["structure"]

    filter_none_python(structure)  # some basic filtering steps

    # filter out test files (unless its pytest)
    if not instance_id.startswith("pytest"):
        filter_out_test_files(structure)

    found_files = []
    found_related_locs = []
    found_edit_locs = []
    additional_artifact_loc_file = None
    additional_artifact_loc_related = None
    additional_artifact_loc_edit_location = None
    file_traj, related_loc_traj, edit_loc_traj = {}, {}, {}

    fl = LLMFL(
        instance_id,
        structure,
        problem_statement,
        args.model,
        args.backend,
        logger,
        args.match_partial_paths,
    )
    # file level localization
    found_files, additional_artifact_loc_file, file_traj = fl.localize(
        mock=args.mock
    )

    # related class, functions, global var localization
    if len(found_files) != 0:
        pred_files = found_files[: args.top_n]
        
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
        

    # Only supports the following args for now
    pred_files = found_files[: args.top_n]
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
    localize_info = {
        "instance_id": instance_id,
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

    with open(args.output_file, "a") as f:
        f.write( json.dumps(localize_info) + "\n")
    return localize_info, structure


def _post_process_multifile_repair(
    raw_output: str,
    file_contents: dict[str, str],
    logger,
    file_loc_intervals: dict[str, list],
    diff_format=False,
):
    edit_multifile_commands = extract_python_blocks(raw_output)
    edited_file = ""
    new_content = ""
    try:
        file_to_commands = split_edit_multifile_commands(
            edit_multifile_commands, diff_format=diff_format
        )
        logger.info("=== file_to_commands: ===")
        logger.info(json.dumps(file_to_commands, indent=2))
        # Let's only edit the first file in the edit commands.
        edited_file_key = next(iter(file_to_commands.keys()))
        logger.info(f"=== edited_file: {edited_file_key} ===")
        edit_commands = file_to_commands[edited_file_key]
        logger.info("=== edit_commands: ===")
        for c in edit_commands:
            logger.info(c)
            logger.info("\n" + "-" * 40)
        edited_file = eval(edited_file_key)  # convert '"file.py"' to 'file.py'
        content = file_contents[edited_file]
        if diff_format:
            new_content = parse_diff_edit_commands(
                edit_commands, content, file_loc_intervals[edited_file]
            )
        else:
            new_content = parse_edit_commands(edit_commands, content)
    except Exception as e:
        logger.error(e)
        return edited_file, new_content

    diff = list(
        unified_diff(
            content.split("\n"),
            new_content.split("\n"),
            fromfile=edited_file,
            tofile=edited_file,
            lineterm="",
        )
    )

    logger.info(f"extracted patch:")
    logger.info("\n".join(diff))
    print("\n".join(diff))
    return edited_file, new_content


def construct_topn_file_context(
    file_to_locs,
    pred_files,
    file_contents,
    structure,
    context_window: int,
    loc_interval: bool = True,
    fine_grain_loc_only: bool = False,
    add_space: bool = False,
    sticky_scroll: bool = False,
    no_line_number: bool = True,
):
    """Concatenate provided locations to form a context.

    loc: {"file_name_1": ["loc_str_1"], ...}
    """
    file_loc_intervals = dict()
    topn_content = ""

    for pred_file, locs in file_to_locs.items():
        content = file_contents[pred_file]
        line_locs, context_intervals = transfer_arb_locs_to_locs(
            locs,
            structure,
            pred_file,
            context_window,
            loc_interval,
            fine_grain_loc_only,
            file_content=file_contents[pred_file] if pred_file in file_contents else "",
        )

        if len(line_locs) > 0:
            # Note that if no location is predicted, we exclude this file.
            file_loc_content = line_wrap_content(
                content,
                context_intervals,
                add_space=add_space,
                no_line_number=no_line_number,
                sticky_scroll=sticky_scroll,
            )
            topn_content += f"### {pred_file}\n{file_loc_content}\n\n\n"
            file_loc_intervals[pred_file] = context_intervals

    return topn_content, file_loc_intervals


def process_loc(localize_info, bench_data, structure, args, logger):
    instance_id = localize_info["instance_id"]
    logger.info(f"================ repairing {instance_id} ================")
    if len(localize_info["found_files"]) == 0:
        # ZZ: TODO fix this return format
        return {
            "instance_id": instance_id,
            "raw_output": [""],
            "try_count": [0],
            "all_generations": [[]],
            "traj": [],
            "prev_content": [[]],
            "file_names": [[]],
        }

    pred_files = localize_info["found_files"][: args.top_n]
    problem_statement = bench_data["problem_statement"]
    files, _, _ = get_full_file_paths_and_classes_and_functions(structure)
    raw_outputs, counts, all_generations, traj, prev_contents, file_names = (
        [],
        [],
        [],
        [],
        [],
        [],
    )

    raw_output = ""
    new_content = ""
    topn_content = ""
    # Construct file contents
    file_contents = dict()
    for i, pred_file in enumerate(pred_files):
        content = None
        for file_content in files:
            if file_content[0] == pred_file:
                content = "\n".join(file_content[1])
                file_contents[pred_file] = content
                break
        assert content is not None, f"{pred_file} file not found"
    # Construct top-n file context
    file_to_edit_locs = dict()
    for i, pred_file in enumerate(pred_files):
        if "found_edit_locs" in localize_info and len(localize_info["found_edit_locs"]) > i:
            file_to_edit_locs[pred_file] = localize_info["found_edit_locs"][i]

    topn_content, file_loc_intervals = construct_topn_file_context(
        file_to_edit_locs,
        pred_files,
        file_contents,
        structure,
        context_window=args.context_window,
        loc_interval=args.loc_interval,
        fine_grain_loc_only=args.fine_grain_loc_only,
        add_space=args.add_space,
        no_line_number=args.diff_format,
        sticky_scroll=args.sticky_scroll,
    )

    if topn_content.strip() == "":
        # ZZ TODO fix this return format
        return {
            "instance_id": instance_id,
            "raw_output": [""],
            "try_count": [0],
            "all_generations": [[]],
            "traj": [],
            "prev_content": [[]],
            "file_names": [[]],
        }

    prompt_template = (
        repair_prompt_combine_topn_cot_diff
        if args.cot and args.diff_format
        else repair_prompt_combine_topn_cot
        if args.cot
        else repair_prompt_combine_topn
    )
    file_instruction = repair_relevant_file_instruction
    message = prompt_template.format(
        repair_relevant_file_instruction=file_instruction,
        problem_statement=problem_statement,
        content=topn_content.rstrip(),
    ).strip()
    logger.info(f"prompting with message:\n{message}")

    all_generations, counts, traj, prev_contents, file_names = [], [], [], [], []
    sample_responses = []
    # Using early stopping will cost more since the input tokens will be charged multiple times.
    # For now we disable it.
    assert args.stop_at_n_unique_valid_samples == -1
    # get temperature samples
    model = make_model(
        model=args.model,
        logger=logger,
        backend=args.backend,
        max_tokens=1024,
        temperature=0.8,
        batch_size=args.max_samples
    )
    sample_trajs = model.codegen(message, num_samples=args.max_samples - 1)
    sample_responses.extend(sample_trajs)

    count = 0
    while count < args.max_samples:
        print(f"trying the {count + 1}-th sample ...")
        ret = sample_responses[count]
        count += 1
        traj.append({**ret, "prompt": message})

        if args.mock:
            continue

        raw_output = ret["response"]
        logger.info(f"raw output:\n{raw_output}")
        all_generations.append(raw_output)

        edited_file, new_content = _post_process_multifile_repair(
            raw_output,
            file_contents,
            logger,
            file_loc_intervals,
            diff_format=args.diff_format,
        )

        if new_content == "":
            prev_contents.append("")
            file_names.append("")
        else:
            prev_content = file_contents[edited_file]
            prev_contents.append(prev_content)
            file_names.append(edited_file)

        counts.append(count)
        raw_outputs.append(raw_output)

    repair_info = {
        "instance_id": instance_id,
        "raw_output": raw_outputs,
        "all_generations": [all_generations],
        "try_count": counts,
        "traj": traj,
        "prev_content": [prev_contents],
        "file_names": [file_names],
    }
    with open(args.output_file, "a") as f:
        f.write(json.dumps(repair_info) + "\n")


def localize_repair_rerank(bench_data, args, existing_instance_ids):
    log_file = os.path.join(args.output_folder, "logs", f"{bench_data['instance_id']}.log")
    logger = setup_logger(log_file)
    localize_info, structure = localize_instance(bench_data, args, existing_instance_ids, logger)
    

def dispatch_tasks(args):
    swe_bench_data = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    existing_instance_ids = (
        load_existing_instance_ids(args.output_file) if args.skip_existing else set()
    )

    if args.num_threads == 1:
        for bench_data in swe_bench_data:
            localize_repair_rerank(bench_data, args, existing_instance_ids)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            futures = [executor.submit(localize_repair_rerank, bench_data, args, 
                        existing_instance_ids) for bench_data in swe_bench_data]
            concurrent.futures.wait(futures)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_folder", type=str, required=True)
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
        default="gpt-4o-mini-2024-07-18",
        choices=["gpt-4o-2024-05-13", "deepseek-coder", "gpt-4o-mini-2024-07-18"],
    )
    parser.add_argument(
        "--backend", type=str, default="openai", choices=["openai", "deepseek"]
    )

    args = parser.parse_args()
    args.loc_output_file = os.path.join(args.output_folder, "location_outputs.jsonl")
    args.rep_output_file = os.path.join(args.output_folder, "repair_outputs.jsonl")
    
    assert (not "deepseek" in args.model) or (
        args.backend == "deepseek"
    ), "Must specify `--backend deepseek` if using a DeepSeek model"

    os.makedirs(os.path.join(args.output_folder, "localization_logs"), exist_ok=True)
    os.makedirs(args.output_folder, exist_ok=True)

    # write the arguments
    with open(f"{args.output_folder}/args.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    dispatch_tasks(args)


if __name__ == "__main__":
    main()
