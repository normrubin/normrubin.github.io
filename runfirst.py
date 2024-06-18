# pre render script 
import os


## copy all the lectures/** files ot slides.***
## change thhe format to reveljs choucklboard = yes
## copy the lectures/*.ipyb to output/_notebooks

## change resource to _noteblooks 


import shutil
import os
import re

def copy_files():
    ''' copy all files from the lectures directory to the slides directory'''
    source_dir = 'lectures'
    target_dir = 'slides'

    # Ensure target directory exists
    os.makedirs(target_dir, exist_ok=True)

    for file_name in os.listdir(source_dir):
        # construct full file path
        source = os.path.join(source_dir, file_name)
        target = os.path.join(target_dir, file_name)
        # copy only files
        if os.path.isfile(source):
            shutil.copy2(source, target)  # copy2 preserves metadata


def set_reveljs():
    directory = 'slides'
    rep = """
format:
    revealjs:
        slide-number: true
        show-slide-number: print
        chalkboard: true
"""

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.qmd'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r+') as f:
                    content = f.read()
                    content_new = re.sub('format: html', rep, content, count=1)
                    if content != content_new:
                        f.seek(0)
                        f.write(content_new)
                        f.truncate()


copy_files()
set_reveljs()
