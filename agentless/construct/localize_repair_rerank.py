import concurrent.futures
import os, json, argparse
from collections import Counter
from difflib import unified_diff
from datasets import load_dataset
from agentless.construct.FL import LLMFL
from agentless.util.model import make_model
from get_repo_structure.get_repo_structure import (get_project_structure_from_scratch)
from agentless.util.utils import (load_existing_instance_ids, setup_logger, parse_git_patch)
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
    normalize_patch,
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
    filter_out_test_files,
    transfer_arb_locs_to_locs,
    get_full_file_paths_and_classes_and_functions,
)


def localize_instance(bench_data, args, logger, patch_info, cache_dir="/dataset/zszeng/AgentlessOutputs/playground"):
    instance_id = bench_data["instance_id"]
    logger.info(f"Processing bug {instance_id}")

    # we need to get the project structure directly
    # ZZ: TODO support multi-thread logic here  
    d = get_project_structure_from_scratch(
        bench_data["repo"], bench_data["base_commit"], instance_id, cache_dir
    )

    logger.info(f"================ localize {instance_id} ================")

    problem_statement = bench_data["problem_statement"]
    structure = d["structure"]

    filter_none_python(structure)  # some basic filtering steps

    # filter out test files (unless its pytest)
    if not instance_id.startswith("pytest"):
        filter_out_test_files(structure)

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
    constructed_hint_traj, constructed_unhint_traj = fl.localize(patch_info, match_partial_paths=True)
    # related class, functions, global var localization
    = fl.localize_function_from_compressed_files(patch_info, constructed_hint_traj, constructed_unhint_traj)

        

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

    with open(args.loc_output_file, "a") as f:
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
    edited_file_list, new_content_list = [], []
    try:
        file_to_commands = split_edit_multifile_commands(
            edit_multifile_commands, diff_format=diff_format
        )
        logger.info("=== file_to_commands: ===")
        logger.info(json.dumps(file_to_commands, indent=2))
        # Let's only edit the first file in the edit commands.
        # ZZ: TODO iterate over all files instead of the first one!
        for edited_file_key in file_to_commands.keys():
            logger.info(f"=== edited_file: {edited_file_key} ===")
            edit_commands = file_to_commands[edited_file_key]
            logger.info("=== edit_commands: ===")
            for c in edit_commands:
                logger.info(c)
                logger.info("\n" + "-" * 40)
            edited_file = eval(edited_file_key)  # convert '"file.py"' to 'file.py'
            content = file_contents[edited_file]
            if diff_format:
                # TODO: Make sure this parse diff format is doing correct thing ....
                new_content = parse_diff_edit_commands(
                    edit_commands, content, file_loc_intervals[edited_file]
                )
            else:
                new_content = parse_edit_commands(edit_commands, content)
            edited_file_list.append(edited_file)
            new_content_list.append(new_content)
    except Exception as e:
        logger.error(e)
        return edited_file_list, new_content_list
    return edited_file_list, new_content_list


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


def repair(localize_info, bench_data, structure, args, logger):
    instance_id = localize_info["instance_id"]
    logger.info(f"================ repairing {instance_id} ================")
    if len(localize_info["found_files"]) == 0:
        return [{
            "model_name_or_path": "agentless",
            "instance_id": instance_id,
            "model_patch": "",
            "raw_model_patch": "",
            "original_file_content": "",
            "try_count": 0,
            "edited_file": "" 
        }]

    pred_files = localize_info["found_files"][: args.top_n]
    problem_statement = bench_data["problem_statement"]
    files, _, _ = get_full_file_paths_and_classes_and_functions(structure)

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
        return [{
            "model_name_or_path": "agentless",
            "instance_id": instance_id,
            "model_patch": "",
            "raw_model_patch": "",
            "original_file_content": "",
            "try_count": 0,
            "edited_file": "" 
        }]
        

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

    # get temperature samples
    model = make_model(
        model=args.model,
        logger=logger,
        backend=args.backend,
        max_tokens=1024,
        temperature=0.8,
        batch_size=args.max_samples
    )
    sample_responses = model.codegen(message, num_samples=args.max_samples)
    repair_info_list = []
    count = 0
    while count < args.max_samples:
        print(f"trying the {count + 1}-th sample ...")
        ret = sample_responses[count]
        count += 1
        curr_traj = {**ret, "prompt": message}

        raw_output = ret["response"]
        logger.info(f"raw output:\n{raw_output}")
        # TODO: WARNING! Ensure if multiple files are within raw_output, they are all processed
        edited_file_list, new_content_list = _post_process_multifile_repair(
            raw_output,
            file_contents,
            logger,
            file_loc_intervals,
            diff_format=args.diff_format,
        )
        raw_git_diffs, normalized_git_diffs = "", ""
        for edited_file, new_content in zip(edited_file_list, new_content_list):
            if edited_file in file_contents:
                content = file_contents[edited_file]
                git_diff = fake_git_repo("playground", edited_file, content, new_content)
                syntax_success = check_syntax(new_content)
                lint_success, prev_errors, errors = lint_code(
                    "playground", "test.py", new_content, file_contents[edited_file]
                )
                differ_by_empty_lines = check_code_differ_by_just_empty_lines(
                    new_content, file_contents[edited_file]
                )
                print(lint_success, prev_errors, errors, differ_by_empty_lines)
                if syntax_success and not differ_by_empty_lines:
                    # ZZ: only add patches that are syntactically correct
                    normalized_patch = normalize_patch(instance_id, git_diff, content)
                    normalized_git_diffs += "\n" + normalized_patch
                    raw_git_diffs += "\n" + git_diff.replace("\ No newline at end of file\n", "")
            else:
                raise f"the edited file is not found: {edited_file}, instance-id: {instance_id} "
        repair_info = {
            "model_name_or_path": "agentless",
            "instance_id": instance_id,
            "model_patch": raw_git_diffs.lstrip(),
            "try_count": count,
            "normalized_patch": normalized_git_diffs,
            "repair_trajectory": curr_traj
        }
        repair_info_list.append(repair_info)
    with open(args.rep_output_file, "a") as f:
        f.write(json.dumps(repair_info_list) + "\n")
    return repair_info_list


def deduplicate_patches(repair_info_list) -> list[str]:
    """Returns all unique patches."""
    patch_keys = [info_obj["normalized_patch"] for info_obj in repair_info_list]
    
    unique_patches = set()
    claned_patches = []
    for i in range(len(patch_keys)):
        patch_key = patch_keys[i].strip()
        if patch_key and patch_key not in unique_patches:
            unique_patches.add(patch_key)
            claned_patches.append(repair_info_list[i])
    return claned_patches


def majority_voting(repair_info_list, args):
    patch_keys = [info_obj["normalized_patch"] for info_obj in repair_info_list]    
    raw_patches = [info_obj["model_patch"] for info_obj in repair_info_list]

    patch_ids = [i for i in range(len(raw_patches)) if raw_patches[i].strip()]
    vote = Counter()
    first_appear_idx = dict()
    for i in patch_ids:
        sample = repair_info_list[i]
        patch_key, patch = sample["normalized_patch"], sample["model_patch"]
        vote[patch_key] += 1
        if patch_key not in first_appear_idx:
            first_appear_idx[patch_key] = i

    maj_selected_id = max(
        patch_ids,
        key=lambda i: (vote[patch_keys[i]], -first_appear_idx[patch_keys[i]]),
    )

    sample = repair_info_list[maj_selected_id]
    result = {
        "model_name_or_path": "agentless",
        "instance_id": sample['instance_id'],
        "model_patch": sample["model_patch"],
        "final_repair_trajectory": sample["repair_trajectory"]
    }
    with open(args.final_output_file, "a") as f:
        f.write(json.dumps(result) + "\n")
    return result



def assemble_trajectories(args, localize_info, repair_info_list, final_result):
    trajectories = {
        "instance_id": localize_info["instance_id"],
        "file_traj": localize_info["file_traj"],
        "related_loc_traj": localize_info["related_loc_traj"],
        "edit_loc_traj": localize_info["edit_loc_traj"],
        "candidate_repair_traj": [info["repair_trajectory"] for info in repair_info_list],
        "selected_repair_traj": final_result["final_repair_trajectory"]
    }
    with open(args.traj_output_file, "a") as f:
        f.write(json.dumps(trajectories)+"\n")
    return trajectories
    

def localize_repair_rerank(bench_data, args, patch_info):
    log_file = os.path.join(args.output_folder, "logs", f"{bench_data['instance_id']}.log")
    logger = setup_logger(log_file)
    localize_info, structure = localize_instance(bench_data, args, logger, patch_info)
    if not localize_info['found_files']:
        logger.warning(f"Instance-{localize_info['instance_id']} fail to localize any files ......")
        return None
    repair_info_list = repair(localize_info, bench_data, structure, args, logger)
    repair_info_list = deduplicate_patches(repair_info_list)
    final_result = majority_voting(repair_info_list, args)
    all_trajectories = assemble_trajectories(args, localize_info, repair_info_list, final_result)
    return all_trajectories
    
    

def dispatch_tasks(args):
    swe_bench_data = load_dataset("princeton-nlp/SWE-bench", split="train")
    existing_instance_ids = (
        load_existing_instance_ids(args.final_output_file) if args.skip_existing else set()
    )
    filtered_bench_data_list, parsed_patch_info_list = [], []
    for bench_data in swe_bench_data:
        if bench_data['instance_id'] in existing_instance_ids: continue # skip existing ids
        patch_info, modified_file_num, invalid_patch_parsing = parse_git_patch(bench_data["patch"])
        if modified_file_num <= 10 and modified_file_num and not invalid_patch_parsing:
            # get rid of invalid patch commits and patches with too many modified files 
            filtered_bench_data_list.append(bench_data)
            parsed_patch_info_list.append(patch_info)
    if args.num_threads == 1:
        for bench_data, patch_info in zip(filtered_bench_data_list, parsed_patch_info_list):
            localize_repair_rerank(bench_data, args, patch_info)
    else:
        # TODO: modify to support multi-thread request
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            futures = [executor.submit(localize_repair_rerank, bench_data, args, 
                        patch_info) for bench_data, patch_info in zip(filtered_bench_data_list, parsed_patch_info_list)]
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
    parser.add_argument("--loc_interval", action="store_true")
    parser.add_argument("--fine_grain_loc_only", action="store_true")
    parser.add_argument("--diff_format", action="store_true")
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--max_samples", type=int, default=20, help="Sampling budget.")
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
        choices=["gpt-4o-2024-05-13", "deepseek-coder", "deepseek-chat", "gpt-4o-mini-2024-07-18"],
    )
    parser.add_argument(
        "--backend", type=str, default="openai", choices=["openai", "deepseek"]
    )

    args = parser.parse_args()
    args.loc_output_file = os.path.join(args.output_folder, "location_outputs.jsonl")
    args.rep_output_file = os.path.join(args.output_folder, "repair_outputs.jsonl")
    args.final_output_file = os.path.join(args.output_folder, "predictions.jsonl")
    args.traj_output_file = os.path.join(args.output_folder, "trajectories.jsonl")
    
    assert (not "deepseek" in args.model) or (
        args.backend == "deepseek"
    ), "Must specify `--backend deepseek` if using a DeepSeek model"

    os.makedirs(os.path.join(args.output_folder, "logs"), exist_ok=True)
    os.makedirs(args.output_folder, exist_ok=True)

    # write the arguments
    with open(f"{args.output_folder}/args.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    dispatch_tasks(args)


if __name__ == "__main__":
    main()
