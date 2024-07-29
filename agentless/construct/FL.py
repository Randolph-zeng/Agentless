from abc import ABC, abstractmethod
from agentless.util.model import make_model
from agentless.repair.repair import construct_topn_file_context
from agentless.util.compress_file import get_skeleton
from agentless.util.postprocess_data import extract_code_blocks, extract_locs_for_files
from agentless.util.preprocess_data import (
    correct_file_paths,
    get_full_file_paths_and_classes_and_functions,
    get_repo_files,
    line_wrap_content,
    show_project_structure,
)
from agentless.construct.prompts import (
    file_content_template,
    obtain_relevant_code_prompt,
    obtain_relevant_files_prompt,
    obtain_relevant_files_prompt_with_hint,
    file_content_in_block_template,
    obtain_relevant_code_combine_top_n_prompt,
    obtain_relevant_functions_from_compressed_files_prompt,
    obtain_relevant_code_combine_top_n_no_line_number_prompt,
    obtain_relevant_functions_and_vars_from_compressed_files_prompt_more
)

MAX_CONTEXT_LENGTH = 128000


class FL(ABC):
    def __init__(self, instance_id, structure, problem_statement, **kwargs):
        self.structure = structure
        self.instance_id = instance_id
        self.problem_statement = problem_statement

    @abstractmethod
    def localize(self, top_n=1, mock=False) -> tuple[list, list, list, any]:
        pass


class LLMFL(FL):
    def __init__(
        self,
        instance_id,
        structure,
        problem_statement,
        model_name,
        backend,
        logger,
        match_partial_paths,
        **kwargs,
    ):
        super().__init__(instance_id, structure, problem_statement)
        self.max_tokens = None
        self.model_name = model_name
        self.backend = backend
        self.logger = logger
        self.match_partial_paths = match_partial_paths

    def _parse_model_return_lines(self, content: str) -> list[str]:
        if content:
            return content.strip().split("\n")

    def _request_file_localization(self, hinted, model, ground_truth_modified_files=None):
        if hinted:
            message = obtain_relevant_files_prompt_with_hint.format(
                problem_statement=self.problem_statement,
                structure=show_project_structure(self.structure).strip(),
                ground_truth_modified_files="\n".join(ground_truth_modified_files)
            ).strip()
        else:    
            message = obtain_relevant_files_prompt.format(
                problem_statement=self.problem_statement,
                structure=show_project_structure(self.structure).strip(),
            ).strip()
        traj = model.codegen(message, num_samples=1)[0]
        traj["prompt"] = message
        # ZZ: use customized parse logic here
        if "Relevant File Paths:" in traj["response"]:
            model_found_files = traj["response"].split("Relevant File Paths:")[-1].strip().split("\n")
        else:
            model_found_files = []
        return model_found_files, traj
    
    def _construct_file_localization_trajectory(self, found_files, ground_truth_modified_files, traj, hinted):
        return {
            "stage": "fault_localization:file_localization",
            "prompt": traj["prompt"],
            "response": traj["response"],
            "predicted_modified_files": found_files,
            "ground_truth_modified_files": ground_truth_modified_files,
            "hinted": hinted
        }

    def localize(
        self, patch_info, match_partial_paths=False
    ) -> tuple[list, list, list, any]:
        ground_truth_modified_files = set([info["original_file_path"] for info in patch_info])
        model = make_model(
            model=self.model_name,
            backend=self.backend,
            logger=self.logger,
            max_tokens=self.max_tokens,
            temperature=0,
            batch_size=1,
        )
        files, classes, functions = get_full_file_paths_and_classes_and_functions(self.structure)
        
        # ZZ: try both hinted and original prompt, but only return found files using the correct ones
        hint_found_files, hint_traj = self._request_file_localization(hinted=True, model=model, ground_truth_modified_files=ground_truth_modified_files)
        unhint_found_files, unhint_traj = self._request_file_localization(hinted=False, model=model, ground_truth_modified_files=None)
        # check if the generated paths actually exist or not
        hint_found_files = correct_file_paths(hint_found_files, files, match_partial_paths=match_partial_paths)
        unhint_found_files = correct_file_paths(unhint_found_files, files, match_partial_paths=match_partial_paths)
        # construct template json  
        constructed_hint_traj = self._construct_file_localization_trajectory( hint_found_files, ground_truth_modified_files, hint_traj, True)
        constructed_unhint_traj = self._construct_file_localization_trajectory( unhint_found_files, ground_truth_modified_files, unhint_traj, False)
        return constructed_hint_traj, constructed_unhint_traj

    def localize_function_for_files(
        self, file_names, mock=False
    ) -> tuple[list, dict, dict]:

        files, classes, functions = get_full_file_paths_and_classes_and_functions(
            self.structure
        )

        max_num_files = len(file_names)
        while 1:
            # added small fix to prevent too many tokens
            contents = []
            for file_name in file_names[:max_num_files]:
                for file_content in files:
                    if file_content[0] == file_name:
                        content = "\n".join(file_content[1])
                        file_content = line_wrap_content(content)
                        contents.append(
                            file_content_template.format(
                                file_name=file_name, file_content=file_content
                            )
                        )
                        break
                else:
                    raise ValueError(f"File {file_name} does not exist.")

            file_contents = "".join(contents)
            if num_tokens_from_messages(file_contents, model) < MAX_CONTEXT_LENGTH:
                break
            else:
                max_num_files -= 1

        message = obtain_relevant_code_combine_top_n_prompt.format(
            problem_statement=self.problem_statement,
            file_contents=file_contents,
        ).strip()
        print(f"prompting with message:\n{message}")
        print("=" * 80)
        if mock:
            self.logger.info("Skipping querying model since mock=True")
            traj = {
                "prompt": message,
                "usage": {
                    "prompt_tokens": num_tokens_from_messages(message, self.model_name),
                },
            }
            return [], {"raw_output_loc": ""}, traj

        model = make_model(
            model=self.model_name,
            backend=self.backend,
            loggger=self.logger,
            max_tokens=self.max_tokens,
            temperature=0,
            batch_size=1,
        )
        traj = model.codegen(message, num_samples=1)[0]
        traj["prompt"] = message
        raw_output = traj["response"]

        model_found_locs = extract_code_blocks(raw_output)
        model_found_locs_separated = extract_locs_for_files(
            model_found_locs, file_names
        )

        print(raw_output)

        return model_found_locs_separated, {"raw_output_loc": raw_output}, traj


    def _construct_function_class_var_localization_message(self, file_names, hinted):
        file_contents = get_repo_files(self.structure, file_names)
        compressed_file_contents = {
            fn: get_skeleton(code) for fn, code in file_contents.items()
        }
        contents = [
            file_content_in_block_template.format(file_name=fn, file_content=code)
            for fn, code in compressed_file_contents.items()
        ]
        file_contents = "".join(contents)
        if hinted:
            pass
        else:            
            message = obtain_relevant_functions_and_vars_from_compressed_files_prompt_more.format(
                problem_statement=self.problem_statement, file_contents=file_contents
            )
        return message
        
    def _request_function_class_var_localization(self, model, message, file_names, ):
        response = model.codegen(message, num_samples=1)[0]
        model_found_locs = extract_code_blocks(response["response"])
        model_found_locs_separated = extract_locs_for_files(
            model_found_locs, file_names
        )
        traj = {
            "stage": "fault_localization:function_class_var_localization",
            "prompt": message,
            "response": response["response"],
            "predicted_modified_files": found_files,
            "ground_truth_modified_files": ground_truth_modified_files,
            "hinted": hinted
        }
        return traj 


    def localize_function_from_compressed_files(self, patch_info, constructed_hint_traj, constructed_unhint_traj):
        model = make_model(
            model=self.model_name,
            backend=self.backend,
            logger=self.logger,
            max_tokens=self.max_tokens,
            temperature=0,
            batch_size=1,
        )
        # TODO: get ground truth function, class, var info etc 
        
        hint_message = self._construct_function_class_var_localization_message(constructed_hint_traj["predicted_modified_files"], True)
        unhint_message = self._construct_function_class_var_localization_message(constructed_unhint_traj["predicted_modified_files"], False)

        self._request_function_class_var_localization()
       
        return xxx

    def localize_line_from_coarse_function_locs(
        self,
        file_names,
        coarse_locs,
        context_window: int,
        add_space: bool,
        sticky_scroll: bool,
        no_line_number: bool,
        temperature: float = 0.0,
        num_samples: int = 1,
        mock=False,
    ):

        file_contents = get_repo_files(self.structure, file_names)
        topn_content, file_loc_intervals = construct_topn_file_context(
            coarse_locs,
            file_names,
            file_contents,
            self.structure,
            context_window=context_window,
            loc_interval=True,
            add_space=add_space,
            sticky_scroll=sticky_scroll,
            no_line_number=no_line_number,
        )
        if no_line_number:
            template = obtain_relevant_code_combine_top_n_no_line_number_prompt
        else:
            template = obtain_relevant_code_combine_top_n_prompt
        message = template.format(
            problem_statement=self.problem_statement, file_contents=topn_content
        )
        self.logger.info(f"prompting with message:\n{message}")
        self.logger.info("=" * 80)
        assert num_tokens_from_messages(message, self.model_name) < MAX_CONTEXT_LENGTH
        if mock:
            self.logger.info("Skipping querying model since mock=True")
            traj = {
                "prompt": message,
                "usage": {
                    "prompt_tokens": num_tokens_from_messages(message, self.model_name),
                },
            }
            return [], {"raw_output_loc": ""}, traj

        model = make_model(
            model=self.model_name,
            backend=self.backend,
            logger=self.logger,
            max_tokens=self.max_tokens,
            temperature=temperature,
            batch_size=num_samples,
        )
        raw_trajs = model.codegen(message, num_samples=num_samples)

        # Merge trajectories
        raw_outputs = [raw_traj["response"] for raw_traj in raw_trajs]
        traj = {
            "prompt": message,
            "response": raw_outputs,
            "usage": {  # merge token usage
                "completion_tokens": sum(
                    raw_traj["usage"]["completion_tokens"] for raw_traj in raw_trajs
                ),
                "prompt_tokens": sum(
                    raw_traj["usage"]["prompt_tokens"] for raw_traj in raw_trajs
                ),
            },
        }
        model_found_locs_separated_in_samples = []
        for raw_output in raw_outputs:
            model_found_locs = extract_code_blocks(raw_output)
            model_found_locs_separated = extract_locs_for_files(
                model_found_locs, file_names
            )
            model_found_locs_separated_in_samples.append(model_found_locs_separated)

            self.logger.info(f"==== raw output ====")
            self.logger.info(raw_output)
            self.logger.info("=" * 80)
            print(raw_output)
            print("=" * 80)
            self.logger.info(f"==== extracted locs ====")
            for loc in model_found_locs_separated:
                self.logger.info(loc)
            self.logger.info("=" * 80)
        self.logger.info("==== Input coarse_locs")
        coarse_info = ""
        for fn, found_locs in coarse_locs.items():
            coarse_info += f"### {fn}\n"
            if isinstance(found_locs, str):
                coarse_info += found_locs + "\n"
            else:
                coarse_info += "\n".join(found_locs) + "\n"
        self.logger.info("\n" + coarse_info)
        if len(model_found_locs_separated_in_samples) == 1:
            model_found_locs_separated_in_samples = (
                model_found_locs_separated_in_samples[0]
            )

        return (
            model_found_locs_separated_in_samples,
            {"raw_output_loc": raw_outputs},
            traj,
        )
