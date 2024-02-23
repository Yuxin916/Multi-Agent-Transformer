import os
import shutil


def clean_empty_model_dirs(root_dir):
    # Check if the directory exists before proceeding
    if not os.path.exists(root_dir):
        print(f"The directory {root_dir} does not exist.")
        return

    # Walk through the subdirectories in the specified root directory
    for subdir, dirs, files in os.walk(root_dir, topdown=False):  # topdown=False to allow deleting
        # For each directory, check if it is named 'run' followed by a number
        if os.path.basename(subdir).startswith('run') and os.path.basename(subdir)[len('run'):].isdigit():
            # Construct the path to the 'models' folder within the 'runX' directory
            models_path = os.path.join(subdir, 'models')
            # Check if the 'models' directory exists
            if os.path.exists(models_path):
                # check if the 'models' directory is empty
                if not os.listdir(models_path):
                    # If the 'models' directory is empty, delete the 'runX' directory
                    shutil.rmtree(subdir)
                    print(f"Deleted {subdir} because 'models' is empty")
                else:
                    # Get all .pt files in the models directory
                    pt_files = [f for f in os.listdir(models_path) if f.endswith('.pt')]
                    # Check if the 'models' directory is empty or has only one .pt file
                    if not pt_files or len(pt_files) <= 1:
                        # If the 'models' directory is empty or has only one .pt file,
                        # delete the 'runX' directory
                        shutil.rmtree(subdir)  # Using shutil.rmtree to delete non-empty dirs
                        print(f"Deleted {subdir} because 'models' has less than one .pt file")
            else:
                shutil.rmtree(subdir)  # Using shutil.rmtree to delete empty dirs
                print(f"Deleted {subdir} because 'models' is empty or has less than one .pt file")

# Usage
# root_directory = './HSN/mat/check'  # Replace with your actual root directory
# root_directory = './StarCraft2_multi/mat/'  # Replace with your actual root directory
# root_directory = './StarCraft2/1c3s5z/mat/check'  # Replace with your actual root directory
# root_directory = './StarCraft2/mat/check'  # Replace with your actual root directory
root_directory = './robotarium/AT/mat/check'  # Replace with your actual root directory

clean_empty_model_dirs(root_directory)
