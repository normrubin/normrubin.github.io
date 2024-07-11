## fix up meta info for all files 
import os 
from pathlib import Path
from turtle import pos
from matplotlib.layout_engine import ConstrainedLayoutEngine
from requests import post
import yaml


def list_all_files(root_dir):
    """List all files in a directory and its subdirectories."""
    all_files = []
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in ['.git', '.venv', "_site", '.quarto']]
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files


files = os.environ.get('QUARTO_PROJECT_INPUT_FILES', False)
if files:
    files = files.split('\n')
else:
    base = Path("/home/norm/compiler_course_2024fa/")
    files = list_all_files(base)



#files = ["test.qmd"]

def compare_dicts(file, dict1, dict2, path=""):
    # Check keys in dict1 not in dict2
    for key in dict1:
        current_path = f"{path}.{key}" if path else key
        if key not in dict2:
            print(f"{file}:Key {current_path} found in the first dictionary but not in the second.")
        elif isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            # If both values are dictionaries, compare recursively
            compare_dicts(file, dict1[key], dict2[key], current_path)
        elif dict1[key] != dict2[key]:
            print(f"{file}: Value for key {current_path} differs: {dict1[key]} vs {dict2[key]}")
    
    # Check keys in dict2 not in dict1
    for key in dict2:
        current_path = f"{path}.{key}" if path else key
        if key not in dict1:
            print(f"{file}: Key {current_path} found in the second dictionary but not in the first.")



def split_file(contents):
    # Find indices of lines that are exactly '---'
    separator_indices = [i for i, line in enumerate(contents) if line.strip() == '---']
    
    # If no separators are found, return the whole contents as pre_contents
    if not separator_indices:
        return contents, "", []
    
    if len(separator_indices) == 1:
        return contents, "", []
    
    # Assuming the first '---' starts the YAML block and the second '---' ends it
    start, end = separator_indices[0], separator_indices[1]
    
    pre_contents = contents[:start+1]
    yaml_contents = "\n".join(contents[start + 1:end])
    post_contents = contents[end:]
    
    return pre_contents, yaml_contents, post_contents

yaml_without_slides = """
title: title
format:
    html: default
"""

yaml_with_slides = """
title: title
format:
    html: default
    revealjs:
        chalkboard: true
        output-file: xxxx
        scrollable: true
        code-line-numbers: true
sidebar: false 
execute:
    echo: true
"""


for file in files:

    #print(file)
    directory, _, file_name = file.rpartition('/')
    yaml_template = yaml.safe_load(yaml_without_slides)
    use_revealjs = False 
    if directory.endswith('lectures'):
        yaml_template = yaml.safe_load(yaml_with_slides)
        use_revealjs = True

    with open(file, 'r') as f:
        try:
            contents = f.read().split('\n')
        except UnicodeDecodeError:
            # print("skip", file)
            continue 
        new_contents = []


        pre, yaml_contents, post  = split_file(contents)
        # Parse the YAML content if any was found
        if yaml_contents:
            yaml_from_file = yaml.safe_load(yaml_contents)
            # preserve some entries 
            keeps = ['title', 'tbl-colwidths']
            for keep in keeps:
                if keep in yaml_from_file:
                    yaml_template[keep] = yaml_from_file[keep]

            if use_revealjs:
                if 'output-location' in yaml_from_file['format']['revealjs']:
                    yaml_template['format']['revealjs']['output-location'] = yaml_from_file['format']['revealjs']['output-location'] 
                yaml_template['format']['revealjs']['output-file'] =  'revealjs_' + file_name
                yaml_from_file['format']['revealjs']['output-file'] = 'revealjs_' + file_name
                yaml_from_file['format']['revealjs']['code-line-numbers'] = 'true'

            compare_dicts(file, yaml_from_file, yaml_template)
    
            #print(file, yaml_template['title'])
        

            new_contents = '\n'.join(pre) + '\n' + yaml.dump(yaml_template) + '\n' + '\n'.join(post)
            with open(file, 'w') as f:
                f.write(new_contents)
