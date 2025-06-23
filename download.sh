#!/bin/bash
pip install gdown tqdm requests urllib3 chardet

# Function to download a file
download_file() {
    local file_id=$1
    local output_file=$2
    echo "Downloading to ${output_file}..."
    gdown "https://drive.google.com/uc?id=${file_id}" -O "${output_file}"
    if [ $? -eq 0 ]; then
        echo "Download complete for ${output_file}!"
    else
        echo "Download failed for ${output_file}. Please try again."
        return 1
    fi
}

OUTPUT_DIR="./"
mkdir -p "$OUTPUT_DIR"


declare -A DIALOGUE_FILES=(
    ["data.zip"]="1fiFsFV-fu94i3szSDjQp3mSdY18b1zKZ" 
)


# Download each dialogue file
for filename in "${!DIALOGUE_FILES[@]}"; do
    file_id="${DIALOGUE_FILES[$filename]}"
    output_file="${OUTPUT_DIR}/${filename}"
    download_file "$file_id" "$output_file"
    # Extract and rename files based on zip name
    unzip -o "$output_file" -d "$OUTPUT_DIR"
    rm -rf "$output_file"
done