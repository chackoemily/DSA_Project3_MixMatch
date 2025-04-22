import pandas as pd
from sklearn.neighbors import NearestNeighbors

def build_graph(songs_csv, threshold):
    df = pd.read_csv(songs_csv)
    track_ids = df["track_id"].astype(str).tolist()

    # song maps
    track_id_to_display_name = {}
    display_name_to_track_id = {}
    for i, track_id in enumerate(track_ids):
        name = df.at[i, "track_name"]
        track_id_to_display_name[track_id] = name
        display_name_to_track_id[name] = track_id

    # quantitative features
    quant_cols = ["duration_ms", "danceability", "energy",
                  "loudness", "valence", "tempo"]
    
    data = df[quant_cols].values 

    track_id_to_index = {track_id: i for i, track_id in enumerate(track_ids)}

    # ball tree to find nearest neighbors
    # Aman said it was okay to use NearestNeighbors and scikitlearn to find the neighbors for the nodes in the graph
    # https://scikit-learn.org/stable/modules/neighbors.html
    nbrs = NearestNeighbors(radius=threshold,
                            metric='manhattan',
                            algorithm='ball_tree',
                            n_jobs=1).fit(data)
    neighbors = nbrs.radius_neighbors_graph(data, mode='distance').tocoo()

    # edge list
    edges = [
        (track_ids[i], track_ids[j], d)
        for i, j, d in zip(neighbors.row, neighbors.col, neighbors.data)
        # so no duplicate edges
        if i < j
    ]

    graph = {}
    
    # create graph from edge list
    for track_id_1, track_id_2, distance in edges:
        graph.setdefault(track_id_1, []).append((track_id_2, distance))
        graph.setdefault(track_id_2, []).append((track_id_1, distance))

    print(f"Loaded {len(track_ids)} songs, {len(edges)} edges")

    return track_id_to_display_name, display_name_to_track_id, graph, data, track_id_to_index
