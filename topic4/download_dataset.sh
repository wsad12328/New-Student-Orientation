# download the dataset from one drive
dataset_url="https://ncku365-my.sharepoint.com/:u:/g/personal/nm6121030_ncku_edu_tw/EbTwo26N_lZOgDCca7v03YwBH-qgc2hpg3hWsHZ_yzgZWQ?e=PseGN4&download=1"
zip_file="cat_dog_kaggle.zip"
dataset_name="cat_dog_kaggle"

# if the dataset is already downloaded, then skip downloading
if [ -f "$zip_file" ]; then
    echo "$zip_file already exists. Skipping download."
else
    echo "Downloading $zip_file..."
    wget -N $dataset_url -O "$zip_file"
fi

# unzip the dataset
if [ -d "$dataset_name" ]; then
    echo "$dataset_name already exists. Skipping unzipping."
else
    echo "Unzipping $zip_file..."
    unzip "$zip_file" -d "$dataset_name"
fi