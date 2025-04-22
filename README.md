# MixMatch

A Python application for generating playlists of similar songs based on two input Spotify tracks. It does this by finding the shortest path between the two input songs using either the Djikstra's or A* path finding algorithm.

## Installation

First, create and activate a virtual environment using Python 3.11, then install the project dependencies:

```bash
# Create a virtual environment named 'venv'
python3.11 -m venv venv

# Activate the environment (macOS/Linux)
source venv/bin/activate

# Windows (PowerShell)
venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

## Dataset

The application expects a cleaned dataset file named `cleaned_dataset.csv` in the project root. You have two options to provide this file:

1. **Use the provided `cleaned_dataset.csv`**

2. **Recreate the dataset yourself**:
   1. Download the raw CSVs from Kaggle:
      - [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
      - [30,000 Spotify Songs](https://www.kaggle.com/datasets/joebeachcapital/30000-spotify-songs)
   2. Place both downloaded CSV files in the project root directory.
   3. The default names of the csvs when downloaded from Kaggle should match what the clean.py code expects.
   4. Run the cleaning script with the virtual environment activated:
      ```bash
      python clean.py
      ```
   5. This will generate `cleaned_dataset.csv` in the root.

## Usage

With the virtual environment activated, run the main application:

```bash
python mixmatch.py
```

> **Note:** The dataset contains tens of thousands of songs and may consume significant RAM. Ensure your machine has enough memory before running the script.
## Additional Modules

This repository also includes two supporting files:

- `build_graph.py`: Code to construct the song graph from `cleaned_dataset.csv` into an adjacency list.
- `algorithms.py`: Implements Dijkstra's and A* pathfinding algorithms.

The code from these modules was integrated into `mixmatch.py`, so you only need to run `mixmatch.py` to use the application. However, you can explore `build_graph.py` and `algorithms.py` if you wish to examine the individual components that makes our project work.

