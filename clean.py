import pandas as pd
import matplotlib.pyplot as plt

def main():
    
    # list of the columns in the datasets that contain the values we need 
    required_cols = [
        "track_id", "track_name", "artists", "duration_ms",
        "danceability", "energy", "loudness", "speechiness",
        "acousticness", "instrumentalness", "valence", "tempo"
    ]
    
    # read datasets
    dataset_1 = pd.read_csv("dataset.csv")
    dataset_2 = pd.read_csv("spotify_songs.csv")
    # rename 'track_artist' column to match first dataset 
    if 'track_artist' in dataset_2.columns:
        dataset_2.rename(columns={'track_artist': 'artists'}, inplace=True)

    # select the data we need from each dataset and combine into one 
    subset_1 = dataset_1[required_cols].copy()
    subset_2 = dataset_2[required_cols].copy()
    combined_dataset = pd.concat([subset_1, subset_2], ignore_index=True)
    
    # remove values that do not make sense and outliers
    combined_dataset = combined_dataset[
        (combined_dataset['duration_ms'] != 0) &
        (combined_dataset['tempo'] != 0) &
        (combined_dataset['duration_ms'] <= 600000)
    ]
    
    combined_dataset.drop_duplicates(subset="track_id", inplace=True)
    
    # list of columns with quantitative data for computing edge weights
    quant_cols = [
        "duration_ms", "danceability", "energy", "loudness",
        "speechiness", "acousticness", "instrumentalness", "valence", "tempo"
    ]
    # normalize all values to be between 0 and 1 
    for i in quant_cols:
        min = combined_dataset[i].min()
        max = combined_dataset[i].max()
        combined_dataset[i] = (combined_dataset[i] - min) / (max - min)
    
    # make histograms for each column to examnine distribution of data
    for i in quant_cols:
        plt.figure()
        plt.hist(combined_dataset[i], bins=50)
        plt.xlabel(i)
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {i}')
        plt.tight_layout()
        hist_file = f"{i}_histogram.png"
        plt.savefig(hist_file)
        plt.close()
        
    # drop the variables with highly skewed/uneven distributions    
    cleaned_dataset = combined_dataset.drop(columns=["acousticness", "speechiness", "instrumentalness"])
    print(f"Total rows in cleaned dataset: {cleaned_dataset.shape[0]}")
    # save cleaned dataset
    cleaned_dataset.to_csv("cleaned_dataset.csv", index=False)


if __name__ == '__main__':
    main()