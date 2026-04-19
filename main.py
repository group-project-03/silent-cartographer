from typing import List, Tuple, Dict
from enum import Enum
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import heapq
import time
from collections import defaultdict

# Actions
class Action(Enum):
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    WAIT = 4

# Turn result
class TurnResult:
    def __init__(self):
        self.wall_hits = 0
        self.current_position = (0, 0)
        self.is_dead = False
        self.is_confused = False
        self.is_goal_reached = False
        self.teleported = False
        self.actions_executed = 0

# object
EMPTY = 0
FIRE = 1
CONFUSION = 2
TP_PURPLE = 3
TP_YELLOW = 4
TP_GREEN = 5

# Image processing
def extract_walls(path: str, grid_size=64, threshold=128):
    img = Image.open(path).convert("RGB")
    arr = np.array(img)

    H, W, _ = arr.shape
    cell_h, cell_w = H / grid_size, W / grid_size

    R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    gray = (np.abs(R - G) < 10) & (np.abs(G - B) < 10)

    lum = 0.299 * R + 0.587 * G + 0.114 * B
    lum[~gray] = 255

    vw = np.zeros((grid_size, grid_size + 1), dtype=int)
    hw = np.zeros((grid_size + 1, grid_size), dtype=int)

    for r in range(grid_size):
        y1, y2 = int(r * cell_h), int((r + 1) * cell_h)
        for c in range(grid_size + 1):
            x = min(max(int(c * cell_w), 0), W - 1)
            vw[r, c] = int(np.mean(lum[y1:y2, x]) < threshold)

    for c in range(grid_size):
        x1, x2 = int(c * cell_w), int((c + 1) * cell_w)
        for r in range(grid_size + 1):
            y = min(max(int(r * cell_h), 0), H - 1)
            hw[r, c] = int(np.mean(lum[y, x1:x2]) < threshold)

    return vw, hw

def extract_objects(path: str, grid_size=64):
    img = Image.open(path).convert("RGB")
    arr = np.array(img)

    H, W, _ = arr.shape
    cell_h, cell_w = H / grid_size, W / grid_size

    obj = np.zeros((grid_size, grid_size), dtype=int)

    for r in range(grid_size):
        for c in range(grid_size):
            y1, y2 = int(r * cell_h), int((r + 1) * cell_h)
            x1, x2 = int(c * cell_w), int((c + 1) * cell_w)

            block = arr[y1:y2, x1:x2]
            R = block[:, :, 0]
            G = block[:, :, 1]
            B = block[:, :, 2]

            # fire
            if np.any((R > 180) & (G > 80) & (G < 190) & (B < 120)):
                obj[r, c] = FIRE

            # Confusion teleport
            elif np.any((R > 170) & (G > 120) & (B < 120)):
                obj[r, c] = CONFUSION

            # purple teleport
            elif np.any((R > 100) & (B > 120) & (G < 120)):
                obj[r, c] = TP_PURPLE

            # yelow star teleport
            elif np.any((R > 180) & (G > 180) & (B < 120)):
                obj[r, c] = TP_YELLOW

            # green star teleport
            elif np.any((G > 160) & (R < 180) & (B < 180)):
                obj[r, c] = TP_GREEN

    return obj

# Environment
class MazeEnvironment:
    def __init__(self, maze_id: str):
        self.maze_id = maze_id
        self.vw, self.hw = extract_walls(maze_id, 64)
        self.obj = extract_objects(maze_id, 64)

        self.size = 64
        self.start, self.goal = self.find_openings()
        self.teleports = self.find_teleports()
        self.reset()

    def find_openings(self):
        openings = []

        for c in range(self.size):
            if self.hw[0, c] == 0:
                openings.append((0, c))
            if self.hw[self.size, c] == 0:
                openings.append((self.size - 1, c))

        for r in range(self.size):
            if self.vw[r, 0] == 0:
                openings.append((r, 0))
            if self.vw[r, self.size] == 0:
                openings.append((r, self.size - 1))

        openings = list(dict.fromkeys(openings))
        if len(openings) < 2:
            return (0, 0), (63, 63)

        return openings[0], openings[-1]

    def find_teleports(self):
        groups = defaultdict(list)

        for r in range(self.size):
            for c in range(self.size):
                t = self.obj[r, c]
                if t in (TP_PURPLE, TP_YELLOW, TP_GREEN):
                    groups[t].append((r, c))

        pairs = {}

        for t, cells in groups.items():
            if len(cells) >= 2:
                for i in range(0, len(cells) - 1, 2):
                    a = cells[i]
                    b = cells[i + 1]
                    pairs[a] = b
                    pairs[b] = a

        return pairs

    def reset(self):
        self.pos = self.start
        self.turns_taken = 0
        self.deaths = 0
        self.confused = 0
        self.path_history = [self.pos]
        self.cells_explored = {self.pos}
        return self.pos

    def neighbors(self, r, c):
        out = []

        if r > 0 and self.hw[r, c] == 0:
            out.append((r - 1, c))
        if r < self.size - 1 and self.hw[r + 1, c] == 0:
            out.append((r + 1, c))
        if c > 0 and self.vw[r, c] == 0:
            out.append((r, c - 1))
        if c < self.size - 1 and self.vw[r, c + 1] == 0:
            out.append((r, c + 1))

        return out

    def step(self, actions: List[Action]):
        result = TurnResult()

        for action in actions[:5]:
            r, c = self.pos
            nr, nc = r, c

            if action == Action.MOVE_UP:
                nr -= 1
            elif action == Action.MOVE_DOWN:
                nr += 1
            elif action == Action.MOVE_LEFT:
                nc -= 1
            elif action == Action.MOVE_RIGHT:
                nc += 1

            if (nr, nc) not in self.neighbors(r, c):
                result.wall_hits += 1
                continue

            self.pos = (nr, nc)
            self.path_history.append(self.pos)
            self.cells_explored.add(self.pos)
            result.actions_executed += 1

            tile = self.obj[self.pos]

            if tile == FIRE:
                self.deaths += 1
                result.is_dead = True

            elif tile == CONFUSION:
                self.confused += 1
                result.is_confused = True

            elif self.pos in self.teleports:
                self.pos = self.teleports[self.pos]
                self.path_history.append(self.pos)
                result.teleported = True

            if self.pos == self.goal:
                result.is_goal_reached = True
                break

        self.turns_taken += 1
        result.current_position = self.pos
        return result


# A* AGENT
class Agent:
    def __init__(self):
        self.path = []
        self.index = 0
        self.env = None

        self.best_score = float("inf")
        self.stable_count = 0
        self.episodes_to_converge = None

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def tile_cost(self, pos):
        t = self.env.obj[pos]
      
        if t == FIRE:
            return 5000          
        elif t == CONFUSION:
            return 20           
        elif t in (TP_PURPLE, TP_YELLOW, TP_GREEN):
            return 1             
        else:
            return 1

    def compute_path(self):
        start = self.env.start
        goal = self.env.goal

        pq = [(0, start)]
        g = {start: 0}
        parent = {}

        while pq:
            _, cur = heapq.heappop(pq)

            if cur == goal:
                break

            for nxt in self.env.neighbors(*cur):
                step_cost = self.tile_cost(nxt)
                new_cost = g[cur] + step_cost
                
                final_nxt = self.env.teleports.get(nxt, nxt)

                if final_nxt not in g or new_cost < g[final_nxt]:
                    g[final_nxt] = new_cost
                    parent[final_nxt] = cur

                    priority = new_cost + self.heuristic(final_nxt, goal)
                    heapq.heappush(pq, (priority, final_nxt))

        if goal not in g:
            self.path = [start]
            return

        node = goal
        path = [goal]

        while node in parent:
            node = parent[node]
            path.append(node)

        self.path = path[::-1]

    def update_learning(self):
        score = len(self.path)

        if score < self.best_score:
            self.best_score = score
            self.stable_count = 1
        elif score == self.best_score:
            self.stable_count += 1
        else:
            self.stable_count = 0

        if self.stable_count >= 3 and self.episodes_to_converge is None:
            self.episodes_to_converge = 3

    def plan_turn(self, last_result):
        if not self.path:
            self.compute_path()

        actions = []

        for _ in range(5):
            if self.index >= len(self.path) - 1:
                break

            r1, c1 = self.path[self.index]
            r2, c2 = self.path[self.index + 1]
            self.index += 1

            if r2 < r1:
                actions.append(Action.MOVE_UP)
            elif r2 > r1:
                actions.append(Action.MOVE_DOWN)
            elif c2 < c1:
                actions.append(Action.MOVE_LEFT)
            else:
                actions.append(Action.MOVE_RIGHT)

        return actions if actions else [Action.WAIT]

    def reset_episode(self):
        self.path = []
        self.index = 0

# Evaluator
class Evaluator:
    def evaluate_agent(self, agent, maze_id, num_episodes=5):
        success = 0
        total_turns = 0
        total_deaths = 0
        total_path_length = 0
        total_unique_cells = 0
        total_visited_cells = 0
        replanning_times = []

        for _ in range(num_episodes):
            env = MazeEnvironment(maze_id)
            agent.env = env
            agent.reset_episode()

            last = None
            t0 = time.time()

            for _ in range(10000):
                actions = agent.plan_turn(last)
                last = env.step(actions)

                if last.is_goal_reached:
                    agent.update_learning()
                    success += 1
                    break

            replanning_times.append(time.time() - t0)

            total_turns += env.turns_taken
            total_deaths += env.deaths
            total_path_length += len(env.path_history)
            total_unique_cells += len(set(env.path_history))
            total_visited_cells += len(env.path_history)

        total_navigable = env.size * env.size

        return {
            "success_rate": (success / num_episodes) * 100,
            "avg_path_length": total_path_length / num_episodes,
            "avg_turns": total_turns / num_episodes,
            "death_rate": total_deaths / total_turns if total_turns else 0,
            "exploration_efficiency":
                total_unique_cells / total_visited_cells,
            "map_completeness":
                total_unique_cells / total_navigable,
            "replanning_efficiency":
                sum(replanning_times) / len(replanning_times),
            "learning_efficiency":
                agent.episodes_to_converge
                if agent.episodes_to_converge is not None
                else "Still Learning"
        }

# Visualizer
class Visualizer:
    def visualize_map(self, env, path=None):
        plt.figure(figsize=(8, 8))
        ax = plt.gca()

        img = Image.open(env.maze_id)
        ax.imshow(img, extent=[0, env.size, env.size, 0])

        if path:
            xs = [p[1] + 0.5 for p in path]
            ys = [p[0] + 0.5 for p in path]
            ax.plot(xs, ys, color="lime", lw=2)

        ax.scatter(env.start[1] + 0.5, env.start[0] + 0.5,
                   color="green", s=80)
        ax.scatter(env.goal[1] + 0.5, env.goal[0] + 0.5,
                   color="blue", s=80)

        ax.set_xlim(0, env.size)
        ax.set_ylim(env.size, 0)
        ax.set_aspect("equal")
        ax.axis("off")
        plt.show()

# main
if __name__ == "__main__":
    maze_file = input("Enter maze image filename: ").strip()

    agent = Agent()
    evaluator = Evaluator()

    results = evaluator.evaluate_agent(agent, maze_file, 5)

    print("\nEvaluation Metrics:")
    for k, v in results.items():
        print(f"{k}: {v}")

    env = MazeEnvironment(maze_file)
    agent.env = env
    agent.compute_path()

    vis = Visualizer()
    vis.visualize_map(env, agent.path)
