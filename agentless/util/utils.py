import os
import re
import json
import logging


def load_jsonl(filepath):
    """
    Load a JSONL file from the given filepath.

    Arguments:
    filepath -- the path to the JSONL file to load

    Returns:
    A list of dictionaries representing the data in each line of the JSONL file.
    """
    with open(filepath, "r") as file:
        return [json.loads(line) for line in file]


def write_jsonl(data, filepath):
    """
    Write data to a JSONL file at the given filepath.

    Arguments:
    data -- a list of dictionaries to write to the JSONL file
    filepath -- the path to the JSONL file to write
    """
    with open(filepath, "w") as file:
        for entry in data:
            file.write(json.dumps(entry) + "\n")


def load_json(filepath):
    return json.load(open(filepath, "r"))


def combine_by_instance_id(data):
    """
    Combine data entries by their instance ID.

    Arguments:
    data -- a list of dictionaries with instance IDs and other information

    Returns:
    A list of combined dictionaries by instance ID with all associated data.
    """
    combined_data = defaultdict(lambda: defaultdict(list))
    for item in data:
        instance_id = item.get("instance_id")
        if not instance_id:
            continue
        for key, value in item.items():
            if key != "instance_id":
                combined_data[instance_id][key].extend(
                    value if isinstance(value, list) else [value]
                )
    return [
        {**{"instance_id": iid}, **details} for iid, details in combined_data.items()
    ]


def setup_logger(log_file):
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    return logger


def load_existing_instance_ids(output_file):
    instance_ids = set()
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    instance_ids.add(data["instance_id"])
                except json.JSONDecodeError:
                    continue
    return instance_ids


def parse_git_patch(patch):
    # ZZ: parse the git patch for ground truth info about fault locations and modifications
    results = []
    # Regex to capture the file names and hunk information
    file_pattern = re.compile(r'diff --git a/(.+) b/(.+)')
    hunk_pattern = re.compile(r'@@ -(\d+),(\d+) \+(\d+),(\d+) @@(?: (.*))?')

    # Split the patch by 'diff --git' to handle each file separately
    file_match_locs = [m for m in re.finditer(file_pattern, patch)]
    modified_file_num = len(file_match_locs)
    invalid_patch_parsing = False
    for f_m_idx, f_match in enumerate(file_match_locs):
        file_start_idx = f_match.start()
        file_end_idx = file_match_locs[f_m_idx+1].start() if (f_m_idx+1) < len(file_match_locs) else None
        modified_file_content = patch[file_start_idx:file_end_idx]
        # get match file path info
        orig_file_path, new_file_path = f_match.groups() 
        # split each modified file by the git patch header
        hunk_match_locs =  [m for m in re.finditer(hunk_pattern, modified_file_content)]
        for h_m_idx, h_match in enumerate(hunk_match_locs):
            hunk_start_idx = h_match.start()
            hunk_end_idx = hunk_match_locs[h_m_idx+1].start() if (h_m_idx+1) < len(hunk_match_locs) else None
            modified_hunk_content = modified_file_content[hunk_start_idx:hunk_end_idx]
            # Get the hunk header info 
            orig_start_line, orig_line_count, new_start_line, new_line_count, hunk_context = re.findall(hunk_pattern, modified_hunk_content)[0]
            # get original and new content
            hunk_lines = modified_file_content[h_match.end(): hunk_end_idx].split('\n')
            orig_hunk_lines, new_hunk_lines = [], []
            for line in hunk_lines:
                if line == '\\ No newline at end of file': continue # ignore the format string added by git
                if line and not line.startswith('+'):
                    if line.startswith('-'):
                        orig_hunk_lines.append(line[1:])
                    else:
                        orig_hunk_lines.append(line)
                if line and not line.startswith('-'):
                    if line.startswith('+'):
                        new_hunk_lines.append(line[1:])
                    else:
                        new_hunk_lines.append(line)
            if not (abs(int(orig_line_count) - len(orig_hunk_lines)) <= 1
                and abs(int(new_line_count) - len(new_hunk_lines)) <=1):
                # This usually happens when the patch itself contains the special git header that messed the re matching
                invalid_patch_parsing = True
            results.append({
                "original_file_path": orig_file_path,
                "new_file_path": new_file_path,
                "original_start_line": orig_start_line,
                "new_start_line": new_start_line,
                "hunk_context": hunk_context,
                "original_hunk_lines": orig_hunk_lines,
                "new_hunk_lines": new_hunk_lines,
                "original_line_count":orig_line_count,
                "new_line_count": new_line_count,
                "hunk_content": modified_hunk_content
            })
    return results, modified_file_num, invalid_patch_parsing