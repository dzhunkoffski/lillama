# lillama

## Install and set the data
```bash
cd data
wget --quiet --show-progress "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
gzip -d TinyStories_all_data.tar.gz
mkdir tiny_stories
cd tiny_stories ; tar -xf TinyStories_all_data.tar
```

