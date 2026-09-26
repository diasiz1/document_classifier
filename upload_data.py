import kagglehub

handle = 'diasiz1/docs_data'
local_dataset_dir = r"C:\Users\izdib\Projects\document_classifier\data"

# Create a new dataset
kagglehub.dataset_upload(handle, local_dataset_dir)

# # You can then create a new version of this dataset and include version notes.
# kagglehub.dataset_upload(handle, local_dataset_dir)

# You can also specify a list of patterns for files/dirs to ignore.
# These patterns are combined with 'kagglehub.datasets.DEFAULT_IGNORE_PATTERNS'
# to determine which files and directories to exclude. 
# To ignore entire directories, include a trailing slash (/) in the pattern.
# kagglehub.dataset_upload(handle, local_dataset_dir)
