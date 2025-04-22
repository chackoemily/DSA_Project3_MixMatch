import time
import heapq
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.popup import Popup
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.scrollview import ScrollView
from kivy.graphics import Color, Rectangle
from kivy.metrics import dp


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

# Defining UI for popup to search for songs
class SearchPopup(Popup):
    def __init__(self, entries, select_callback, **kwargs):
        super().__init__(**kwargs)

        # Overall layout
        self.title = "Search Song"
        self.size_hint = (0.8, 0.8)
        self.entries = entries
        self.select_callback = select_callback

        # Search input
        root = BoxLayout(orientation='vertical', padding=10, spacing=10)
        self.search_input = TextInput(
            size_hint_y=None, height=40, multiline=False,
            hint_text="Type at least 2 characters",
            foreground_color=(0, 0, 0, 1),
            background_normal='', background_active=''
        )
        self.search_input.bind(text=self.on_text)
        self.search_input.bind(on_text_validate=lambda inst: self.on_text(inst, inst.text))
        root.add_widget(self.search_input)


        # Scroll view for all the results
        self.results_container = BoxLayout(
            orientation='vertical',
            size_hint_y=None,
            spacing=2,
            padding=(0, 2)
        )
        self.results_container.bind(
            minimum_height=self.results_container.setter('height')
        )
        scroll = ScrollView(size_hint=(1, 1))
        scroll.add_widget(self.results_container)
        root.add_widget(scroll)

        self.content = root

    # Function that is called everytime search input changes
    def on_text(self, _, text):
        # Clear previous results
        self.results_container.clear_widgets()

        # Do not search for less than 2 characters
        if len(text) < 2:
            return

        # Case-insensitive substring match
        matches = [
            e for e in self.entries
            if text.lower() in e.lower()
        ][:100]


        # Shows no results if no results found
        if not matches:
            self.results_container.add_widget(Label(
                text="No songs found",
                size_hint_y=None, height=dp(40),
                color=(1, 0, 0, 1), halign='center', valign='middle'
            ))
            return

        # Put the results in the resutls container and make them all clickable
        for name in matches:
            btn = Button(
                text=name,
                size_hint_y=None, height=dp(44),
                background_normal='', background_down='',
                background_color=(1, 1, 1, 1),
                color=(0, 0, 0, 1)
            )
            btn.bind(on_release=lambda btn, n=name: self._select(n))
            self.results_container.add_widget(btn)

    # Calling callback with song name and closing modal on select
    def _select(self, name):
        self.select_callback(name)
        self.dismiss()

# Makes the background green
class ColoredBackground(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.canvas.before:
            Color(0.7098, 0.9725, 0.8039, 1)  # mint‑green
            self.rect = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=self._upd, size=self._upd)

    # Updates background dimensions if you change the size of hte window
    def _upd(self, inst, _):
        self.rect.pos = inst.pos
        self.rect.size = inst.size

# https://kivy.org/doc/stable/guide/basic.html
# introductory guide: https://www.geeksforgeeks.org/python-make-a-simple-window-using-kivy/
class MixMatchApp(App):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.track_id_to_display_name = {}
        self.display_name_to_track_id = {}
        self.graph = {}
        self.display_list = []
        self.data = None
        self.track_id_to_index = {}

    # Runs on startup to initialize graph and other variables
    def startup_function(self):
        (self.track_id_to_display_name,
         self.display_name_to_track_id,
         self.graph,
         self.data,
         self.track_id_to_index) = build_graph(
            "cleaned_dataset.csv", 0.25)

        # fill search entries
        self.display_list = [str(name) for name, _ in self.display_name_to_track_id.items()]

    # Overloading build function
    # Most of the components are explained here: https://kivy.org/doc/stable/api-kivy.uix.html
    def build(self):
        # Run startup function
        self.startup_function()

        # Apply thge green background
        root = ColoredBackground(orientation='horizontal',
                                 padding=20, spacing=20)

        # Defining left column of UI
        # https://kivy.org/doc/stable/gettingstarted/layouts.html
        left = AnchorLayout(anchor_x='left', anchor_y='top',
                            size_hint_x=0.4)
        bounding_box = BoxLayout(orientation='vertical',
                         size_hint=(1, None), spacing=15)
        bounding_box.bind(minimum_height=bounding_box.setter('height'))


        # Function for making sub-title text consistently
        def make_text(text, height=30):
            label = Label(text=text, size_hint_y=None, height=height,
                      halign='left', valign='middle',
                      color=(0, 0, 0, 1))
            label.bind(size=lambda inst, val:
                   setattr(inst, 'text_size', (inst.width, inst.height)))
            return label

        # Add all the text and buttons in left column
        bounding_box.add_widget(make_text("MixMatch", 40))
        bounding_box.add_widget(make_text("Pick a start song:"))
        self.start_button = Button(text="Select start…",
                                size_hint_y=None, height=dp(44))
        self.start_button.bind(on_release=lambda *a:
                            self.search('start'))
        bounding_box.add_widget(self.start_button)

        bounding_box.add_widget(make_text("Pick an end song:"))
        self.end_button = Button(text="Select end…",
                              size_hint_y=None, height=dp(44))
        self.end_button.bind(on_release=lambda *a:
                          self.search('end'))
        bounding_box.add_widget(self.end_button)

        bounding_box.add_widget(make_text("Pick an algorithm:"))
        algo_selection = BoxLayout(size_hint_y=None,
                              height=dp(44), spacing=10)
        self.djikstras_selected = ToggleButton(text="Dijkstra's",
                                       group='alg', state='down')
        self.astar_selected = ToggleButton(text="A*",
                                         group='alg')
        
        algo_selection.add_widget(self.djikstras_selected)
        algo_selection.add_widget(self.astar_selected)
        bounding_box.add_widget(algo_selection)

        # Find button which calls the selected algorithm
        find_button = Button(text="Find Connection",
                          size_hint_y=None, height=dp(50))
        find_button.bind(on_release=self.on_find)
        bounding_box.add_widget(find_button)

        left.add_widget(bounding_box)
        root.add_widget(left)

        # Defining right column of UI which displays results
        right_anchor = AnchorLayout(
            anchor_x='right',
            anchor_y='top',
            size_hint_x=0.6
        )
        self.results_box = BoxLayout(
            orientation='vertical',
            spacing=10,
            size_hint=(1, None)
        )
        self.results_box.bind(minimum_height=self.results_box.setter('height'))
        right_anchor.add_widget(self.results_box)
        root.add_widget(right_anchor)

        return root

    
    def search(self, field):
        def on_select(name):
            if field == 'start':
                self.start_button.text = name
                self.start_id = self.display_name_to_track_id[name]
            else:
                self.end_button.text = name
                self.end_id = self.display_name_to_track_id[name]

        popup = SearchPopup(self.display_list, on_select)
        popup.open()

    def on_find(self, *args):
        # choose algorithm
        algo = self.dijkstra if self.djikstras_selected.state == 'down' else self.astar

        # get execution time
        start_time = time.time()
        dist, path = algo(self.start_id, self.end_id)
        time_taken = time.time() - start_time

        # helper to wrap text to multiple lines
        def make_wrapped(text, min_height=dp(40)):
            label = Label(
                text=text,
                color=(0, 0, 0, 1),
                size_hint_y=None,
                halign='left',
                valign='top'
            )
            # Wrap text to the label's width
            label.bind(width=lambda instance, width: instance.setter('text_size')(instance, (width, None)))
            # Adjust height to texture
            label.bind(texture_size=lambda instance, texture_size: setattr(instance, 'height', max((texture_size[1], min_height))))
            return label

        # Clear previous results
        self.results_box.clear_widgets()

        # Show results
        if path is None:
            # Show 'no path found' if no path found
            self.results_box.add_widget(make_wrapped("No path found", min_height=dp(30)))
        else:
            # Map the ids back to songs and connect them with arrows
            names = [self.track_id_to_display_name[tid] for tid in path]
            path = " -> ".join(names)
            # Wrap the path string because it can be long
            self.results_box.add_widget(make_wrapped(path))

            # Show the total distance of the path
            self.results_box.add_widget(Label(
                text=f"Distance: {dist}",
                color=(0, 0, 0, 1),
                size_hint_y=None,
                height=dp(30),
                halign='left',
                valign='middle'
            ))

        # Show how long it took to execute the path finding algorithm
        self.results_box.add_widget(Label(
            text=f"Time: {time_taken:.4f} s",
            color=(0, 0, 0, 1),
            size_hint_y=None,
            height=dp(30),
            halign='left',
            valign='middle'
        ))


    def dijkstra(self, start, goal):
        # priority queue/min heap of (distances so far, node); initialize with start
        distance_heap = [(0, start)]

        # stores the best known distance to each node
        best_dist = {start: 0}

        # backpointers to previous nodes on the shortest path
        prev_nodes = {}
        
        while distance_heap:
            current_dist, current_node = heapq.heappop(distance_heap)
            
            # If we already reached the end point
            if current_node == goal:
                break

            # Doesn't explore a path longer than our shortest recorded path
            if current_dist > best_dist[current_node]:
                continue

            # explores each neighbor of the current node
            for neighbor, weight in self.graph.get(current_node, []):
                new_distance = current_dist + weight
                # if this new path is shorter, records it
                if new_distance < best_dist.get(neighbor, float('inf')):
                    best_dist[neighbor] = new_distance
                    prev_nodes[neighbor] = current_node
                    heapq.heappush(distance_heap, (new_distance, neighbor))

        # if there is no path from start to end
        if goal not in best_dist:
            return [], float('inf')

        # makes a path from previous nodes backwards
        path = []
        current = goal
        while current != start:
            path.append(current)
            current = prev_nodes[current]
        path.append(start)
        path.reverse()

        return best_dist[goal], path

    # https://www.geeksforgeeks.org/a-search-algorithm/
    def astar(self, start, goal):
        # Manhattan distance heuristic with the track features
        def heuristic(track_id_1, track_id_2):
            i1 = self.track_id_to_index[track_id_1]
            i2 = self.track_id_to_index[track_id_2]
            return np.sum(np.abs(self.data[i1] - self.data[i2]))

        # min heap of f_Score = g + h, node
        f_heap = [(heuristic(start, goal), start)]
        # stores best known cost from start to each node
        g_score = {start: 0}
        # set of nodes already seen
        closed_set = set()
        # set of previous nodes to make final path
        came_from = {}

        # djikstra's with heuristic added
        while f_heap:
            f, node = heapq.heappop(f_heap)
            if node == goal:
                path = []
                cur = node
                while cur != start:
                    path.append(cur)
                    cur = came_from[cur]
                path.append(start)
                path.reverse()
                return g_score[node], path

            closed_set.add(node)
            for nbr, w in self.graph.get(node, []):
                if nbr in closed_set:
                    continue
                tg = g_score[node] + w
                if tg < g_score.get(nbr, float('inf')):
                    g_score[nbr] = tg
                    came_from[nbr] = node
                    heapq.heappush(f_heap, (tg + heuristic(nbr, goal), nbr))

        return None, None

if __name__ == '__main__':
    MixMatchApp().run()
