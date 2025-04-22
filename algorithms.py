import heapq
import numpy as np

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