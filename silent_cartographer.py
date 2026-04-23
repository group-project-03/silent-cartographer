import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import heapq

def detect_flame_pivots(env):
    N = env.size
    hz = env.hz

    pivots = []

    diag_dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    def in_bounds(r, c):
        return 0 <= r < N and 0 <= c < N

    for r in range(N):
        for c in range(N):

            if hz[r, c] != 1:
                continue

            diag_hits = []
            for dr, dc in diag_dirs:
                nr, nc = r + dr, c + dc
                if in_bounds(nr, nc) and hz[nr, nc] == 1:
                    diag_hits.append((dr, dc))

            if len(diag_hits) < 2:
                continue

            is_pivot = False
            for i in range(len(diag_hits)):
                for j in range(i + 1, len(diag_hits)):
                    d1 = diag_hits[i]
                    d2 = diag_hits[j]

                    if (d1[0] == -d2[0]) and (d1[1] == -d2[1]):
                        continue  

                    is_pivot = True
                    break
                if is_pivot:
                    break

            if is_pivot:
                pivots.append((r, c))

    env.flame_pivots = pivots


def extract_walls(path, grid_size=64, threshold=128):
    img = Image.open(path).convert('RGB')
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

def extract_hazards(path, grid_size=64):
    img = Image.open(path).convert('RGB')
    arr = np.array(img)
    H, W, _ = arr.shape

    cell_h, cell_w = H / grid_size, W / grid_size

    R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    gray = (np.abs(R - G) < 10) & (np.abs(G - B) < 10)

    hz = np.zeros((grid_size, grid_size), dtype=int)

    for r in range(grid_size):
        y1, y2 = int(r * cell_h), int((r + 1) * cell_h)
        for c in range(grid_size):
            x1, x2 = int(c * cell_w), int((c + 1) * cell_w)
            block = ~gray[y1:y2, x1:x2]
            hz[r, c] = int(np.any(block))

    return hz

def extract_white_mask(path, grid_size=64,
                       color_threshold=30,
                       ratio_threshold=0.90):
    img = Image.open(path).convert('RGB')
    arr = np.array(img)
    H, W, _ = arr.shape

    cell_h, cell_w = H / grid_size, W / grid_size

    R = arr[:, :, 0].astype(float)
    G = arr[:, :, 1].astype(float)
    B = arr[:, :, 2].astype(float)

    color_dist = np.sqrt((R - G)**2 + (G - B)**2 + (B - R)**2)

    white_mask = np.zeros((grid_size, grid_size), dtype=int)

    for r in range(grid_size):
        y1, y2 = int(r * cell_h), int((r + 1) * cell_h)
        for c in range(grid_size):
            x1, x2 = int(c * cell_w), int((c + 1) * cell_w)

            block = color_dist[y1:y2, x1:x2]

            white_pixels = np.sum(block < color_threshold)
            total_pixels = block.size
            white_ratio = white_pixels / total_pixels

            white_mask[r, c] = 0 if white_ratio >= ratio_threshold else 1

    return white_mask

class MazeEnvironment:
    def __init__(self, vw, hw, hz, white_mask):
        self.vw = vw
        self.hw = hw
        self.hz = hz
        self.white_mask = white_mask
        self.size = vw.shape[0]

        self.hazard_labels = np.full((self.size, self.size), "", dtype=object)
        self.flame_pivots = []

    @classmethod
    def from_image(cls, path, grid_size=64):
        vw, hw = extract_walls(path, grid_size)
        hz = extract_hazards(path, grid_size)
        white_mask = extract_white_mask(path, grid_size)
        return cls(vw, hw, hz, white_mask)

    def neighbors(self, r, c):
        N = self.size
        out = []
        if r > 0 and self.hw[r, c] == 0:
            out.append((r - 1, c))
        if r < N - 1 and self.hw[r + 1, c] == 0:
            out.append((r + 1, c))
        if c > 0 and self.vw[r, c] == 0:
            out.append((r, c - 1))
        if c < N - 1 and self.vw[r, c + 1] == 0:
            out.append((r, c + 1))
        return out

def find_openings(env):
    N = env.size
    op = []

    for c in range(N):
        if env.hw[0, c] == 0:
            op.append((0, c))
        if env.hw[N, c] == 0:
            op.append((N - 1, c))

    for r in range(N):
        if env.vw[r, 0] == 0:
            op.append((r, 0))
        if env.vw[r, N] == 0:
            op.append((r, N - 1))

    return list(dict.fromkeys(op))

class HazardCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256), nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def train_hazard_cnn(data_dir="hazards", model_path="hazard_cnn.pth",
                     batch_size=32, epochs=10, lr=1e-3):

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = HazardCNN(num_classes=len(dataset.classes))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    print("Training CNN on hazard icons...")
    for epoch in range(epochs):
        total = 0
        correct = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            opt.zero_grad()
            out = model(imgs)
            loss = loss_fn(out, labels)
            loss.backward()
            opt.step()

            pred = out.argmax(1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

        print(f"Epoch {epoch+1}/{epochs}  Acc={correct/total:.3f}")

    torch.save({
        "model": model.state_dict(),
        "classes": dataset.classes
    }, model_path)

    print("Saved CNN to", model_path)
    return model, dataset.classes

def load_hazard_cnn(model_path="hazard_cnn.pth"):
    ckpt = torch.load(model_path, map_location="cpu")
    classes = ckpt["classes"]

    model = HazardCNN(num_classes=len(classes))
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, classes

def classify_hazards(env, image_path, model, classes):
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)
    H, W, _ = arr.shape
    N = env.size
    cell_h, cell_w = H / N, W / N

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    env.hazard_labels = np.full((N, N), "", dtype=object)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    for r in range(N):
        for c in range(N):
            if env.hz[r, c] == 1:
                y1, y2 = int(r * cell_h), int((r + 1) * cell_h)
                x1, x2 = int(c * cell_w), int((c + 1) * cell_w)

                crop = Image.fromarray(arr[y1:y2, x1:x2])
                tensor = transform(crop).unsqueeze(0).to(device)

                with torch.no_grad():
                    out = model(tensor)
                    pred = out.argmax(1).item()

                env.hazard_labels[r, c] = classes[pred]

def rotate_flame_state(pivot, arms):
    pr, pc = pivot
    cw = {
        (-1, -1): (-1, +1),
        (-1, +1): (+1, +1),
        (+1, +1): (+1, -1),
        (+1, -1): (-1, -1)
    }

    new_positions = []

    for (dr, dc) in arms:
        if (dr, dc) not in cw:
            new_positions.append((pr + dr, pc + dc))
        else:
            ndr, ndc = cw[(dr, dc)]
            new_positions.append((pr + ndr, pc + ndc))

    return new_positions



def invert_direction(move):
    inv = {
        "UP": "DOWN",
        "DOWN": "UP",
        "LEFT": "RIGHT",
        "RIGHT": "LEFT"
    }
    return inv.get(move, move)


def teleport_circle(env, r, c):
    return env.circle_pairs.get((r, c), (r, c))


def shift_left(r, c):
    return (r, c - 1)


def shift_up(r, c):
    return (r - 1, c)

class Agent:
    def __init__(self, env):
        self.env = env
        N = env.size

        self.known_vw = -np.ones_like(env.vw)
        self.known_hw = -np.ones_like(env.hw)
        self.known_hz = -np.ones_like(env.hz)
        self.known_white = -np.ones_like(env.white_mask)
        self.known_labels = np.full((N, N), "", dtype=object)
        self.known_pivots = set()

        self.visited = set()
        self.replans = 0
        self.deaths = 0
        self.q_influence = QInfluence() 

    def sense(self, r, c):
        self.visited.add((r, c))

        self.known_hz[r, c] = self.env.hz[r, c]
        self.known_white[r, c] = self.env.white_mask[r, c]
        self.known_labels[r, c] = self.env.hazard_labels[r, c]

        if (r, c) in self.env.flame_pivots:
            self.known_pivots.add((r, c))

        if r > 0:
            self.known_hw[r, c] = self.env.hw[r, c]
        if r < self.env.size - 1:
            self.known_hw[r + 1, c] = self.env.hw[r + 1, c]
        if c > 0:
            self.known_vw[r, c] = self.env.vw[r, c]
        if c < self.env.size - 1:
            self.known_vw[r, c + 1] = self.env.vw[r, c + 1]

    def neighbors(self, r, c):
        N = self.env.size
        out = []

        if r > 0 and self.known_hw[r, c] != 1:
            out.append((r - 1, c))
        if r < N - 1 and self.known_hw[r + 1, c] != 1:
            out.append((r + 1, c))
        if c > 0 and self.known_vw[r, c] != 1:
            out.append((r, c - 1))
        if c < N - 1 and self.known_vw[r, c + 1] != 1:
            out.append((r, c + 1))

        return out
    
    def apply_cell_effects(self, cell, move=None):
        r, c = cell
        label = self.env.hazard_labels[r, c].lower()

        result = {
            "inverted_move": False,
            "teleport_to": None,
            "shift_left": False,
            "shift_up": False
        }

        if (r, c) in self.env.confused_cells or label == "confused":
            result["inverted_move"] = True

        if (r, c) in self.env.circle_pairs or label == "circle":
            result["teleport_to"] = teleport_circle(self.env, r, c)

        if (r, c) in self.env.left_shift_cells or label == "left":
            result["shift_left"] = True

        if (r, c) in self.env.up_shift_cells or label == "up":
            result["shift_up"] = True

        return result


    def belief_cost(self, cell):
        r, c = cell

        hz = self.known_hz[r, c]
        white = self.known_white[r, c]
        label = self.known_labels[r, c]
        is_pivot = (r, c) in self.known_pivots

        cost = 1

        if is_pivot:
            cost += 10000
        elif label.lower() == "flame":
            cost += 8000
        elif white == 1 and hz == 1 and label == "":
            cost += 1000

        if hz == 1:
            cost += 2000

        return cost

    def astar(self, start, goal):
        self.replans += 1

        def h(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        pq = [(0, start)]
        g = {start: 0}
        parent = {}

        while pq:
            _, cur = heapq.heappop(pq)

            if cur == goal:
                path = [cur]
                while cur in parent:
                    cur = parent[cur]
                    path.append(cur)
                return path[::-1]

            for nxt in self.neighbors(*cur):
                base_cost = self.belief_cost(nxt)

                d_cur = abs(cur[0] - goal[0]) + abs(cur[1] - goal[1])
                d_nxt = abs(nxt[0] - goal[0]) + abs(nxt[1] - goal[1])

                reward_strength = 10
                proximity_reward = reward_strength * (d_cur - d_nxt)
                q_bias = self.q_influence.get_bias(cur, nxt)
                cost = g[cur] + base_cost - proximity_reward - q_bias
                if nxt not in g or cost < g[nxt]:
                    g[nxt] = cost
                    parent[nxt] = cur
                    heapq.heappush(pq, (cost + h(nxt, goal), nxt))

        return None

    def is_deadly(self, cell):
        r, c = cell
        label = self.env.hazard_labels[r, c].lower()
        is_pivot = (r, c) in self.env.flame_pivots
        return is_pivot or label == "flame"

    def solve_episodes(self, start, goal, max_episodes=5):
        all_paths = []
        episode_lengths = []

        current_start = start

        for episode in range(max_episodes):

            current = current_start
            episode_path = [current]
            self.sense(*current)

            while current != goal:
                path = self.astar(current, goal)

                if not path or len(path) < 2:
                    print("No path possible with discovered map.")
                    return all_paths, episode_lengths

                nxt = path[1]
                self.sense(*nxt)

                if self.is_deadly(nxt):
                    self.deaths += 1

                if nxt not in self.env.neighbors(*current):
                    continue

                current = nxt
                episode_path.append(current)

            all_paths.append(episode_path)
            episode_lengths.append(len(episode_path))

            current_start = start

        return all_paths, episode_lengths

class QInfluence:
    """
    Inactive module that provides Q-value based biasing for A*.
    It does NOT affect the agent until enabled.
    """

    ACTIONS = {
        ( -1,  0): "UP",
        ( +1,  0): "DOWN",
        (  0, -1): "LEFT",
        (  0, +1): "RIGHT"
    }

    def __init__(self):
        self.enabled = True
        self.Q = {} 

    def load_qtable(self, qtable):
        if not qtable or not isinstance(qtable, dict):
            self.Q = {}
            self.enabled = False
            return

        self.Q = qtable


    def get_bias(self, cur, nxt):
        if not self.enabled:
            return 0.0

        r, c = cur
        nr, nc = nxt
        dr, dc = nr - r, nc - c

        action = self.ACTIONS.get((dr, dc), None)
        if action is None:
            return 0.0

        state_key = (r, c)
        if state_key not in self.Q:
            return 0.0

        return self.Q[state_key].get(action, 0.0)


def render(env, path=None, start=None, goal=None,
           save='maze_render.png', background_image=None, agent=None):
    plt.figure(figsize=(8, 8))
    ax = plt.gca()

    N = env.size

    if background_image:
        img = Image.open(background_image).convert('RGB')
        arr = np.array(img)
        H, W, _ = arr.shape
        cell_h, cell_w = H / N, W / N

        ax.imshow(arr, extent=[0, N, N, 0])

    ax.set_aspect('equal')
    ax.invert_yaxis()

    if agent is not None:
        for r, c in agent.visited:
            ax.add_patch(plt.Rectangle(
                (c, r), 1, 1,
                facecolor='cyan', alpha=0.15
            ))

    for r in range(N):
        for c in range(N + 1):
            if env.vw[r, c] == 1:
                ax.plot([c, c], [r, r + 1], color='black', lw=1)

    for r in range(N + 1):
        for c in range(N):
            if env.hw[r, c] == 1:
                ax.plot([c, c + 1], [r, r], color='black', lw=1)

    for r in range(N):
        for c in range(N):
            if env.white_mask[r, c] == 1:
                ax.add_patch(plt.Rectangle(
                    (c, r), 1, 1,
                    facecolor='red', alpha=0.25
                ))

    if background_image:
        for r in range(N):
            for c in range(N):
                if env.hz[r, c] == 1:
                    y1, y2 = int(r * cell_h), int((r + 1) * cell_h)
                    x1, x2 = int(c * cell_w), int((c + 1) * cell_w)
                    icon = arr[y1:y2, x1:x2]
                    ax.imshow(icon, extent=[c, c + 1, r + 1, r])
    """
    if hasattr(env, "hazard_labels"):
        for r in range(N):
            for c in range(N):
                label = env.hazard_labels[r, c]
                if label != "":
                    ax.text(c + 0.5, r + 0.5, label,
                            color="gray", fontsize=6,
                            ha="center", va="center",
                            fontweight="bold")
    """
    if hasattr(env, "flame_pivots"):
        for r, c in env.flame_pivots:
            ax.text(
                c + 0.5, r + 0.5, "P",
                color="yellow",
                fontsize=10,
                ha="center",
                va="center",
                fontweight="bold"
            )
            ax.scatter(c + 0.5, r + 0.5, s=80, c="yellow", edgecolors="black")

    if path:
        xs = [p[1] + 0.5 for p in path]
        ys = [p[0] + 0.5 for p in path]
        ax.plot(xs, ys, lw=2, color="blue")

    if start is not None:
        ax.scatter(start[1] + 0.5, start[0] + 0.5, s=80, c="green")
    if goal is not None:
        ax.scatter(goal[1] + 0.5, goal[0] + 0.5, s=80, c="orange")

    ax.set_xlim(0, N)
    ax.set_ylim(N, 0)
    ax.axis('off')
    plt.savefig(save, bbox_inches='tight', dpi=200)
    plt.show()

class Evaluator:
    def __init__(self, maze_path, model, classes, grid_size=64):
        self.maze_path = maze_path
        self.model = model
        self.classes = classes
        self.grid_size = grid_size

    def evaluate_agent(self, num_episodes=5):
        success = 0
        total_turns = 0
        total_deaths = 0
        total_path_length = 0
        total_unique_cells = 0
        total_visited_cells = 0
        replanning_times = []

        best_turns = float("inf")
        best_path = float("inf")

        env = MazeEnvironment.from_image(self.maze_path, self.grid_size)
        classify_hazards(env, self.maze_path, self.model, self.classes)
        detect_flame_pivots(env)

        agent = Agent(env)

        openings = find_openings(env)
        start, goal = openings[0], openings[-1]

        for episode in range(num_episodes):

            import time
            t0 = time.time()

            all_paths, episode_lengths = agent.solve_episodes(start, goal, max_episodes=5)

            replanning_times.append(time.time() - t0)

            if not all_paths:
                continue

            path = all_paths[0]
            steps = len(path) - 1

            success += 1

            best_turns = min(best_turns, steps)
            best_path = min(best_path, len(path))

            total_turns += steps
            total_path_length += len(path)
            total_deaths += agent.deaths
            total_unique_cells = len(agent.visited)
            total_visited_cells += len(path)

        total_navigable = env.size * env.size
        

        return {
            "success_rate": (success / num_episodes) * 100,

            "avg_path_length": total_path_length / num_episodes if num_episodes else 0,
            "avg_turns": total_turns / num_episodes if num_episodes else 0,

            "best_path_length": best_path if success > 0 else "No Success",
            "best_turns": best_turns if success > 0 else "No Success",

            "death_rate": total_deaths / total_turns if total_turns else 0,

            "exploration_efficiency":
                total_unique_cells / total_visited_cells
                if total_visited_cells else 0,

            "map_completeness":
                total_unique_cells / total_navigable,

            "replanning_efficiency":
                sum(replanning_times) / len(replanning_times),

            "learning_efficiency, for training purposes":
                success   
        }

if __name__ == '__main__':
    file = input('Enter maze image: ').strip()
    grid_size = 64

    env = MazeEnvironment.from_image(file, grid_size)

    if not os.path.exists("hazard_cnn.pth"):
        model, classes = train_hazard_cnn("hazards")
    else:
        model, classes = load_hazard_cnn("hazard_cnn.pth")

    classify_hazards(env, file, model, classes)
    detect_flame_pivots(env)

    print("Flame pivots found:", env.flame_pivots)
    print("Maze size:", env.size)
    print("Openings:", find_openings(env))

    openings = find_openings(env)
    if len(openings) < 2:
        print("Not enough openings found.")
    else:
        start, goal = openings[0], openings[-1]
        agent = Agent(env)

        all_paths, episode_lengths = agent.solve_episodes(start, goal, max_episodes=5)

        print("Episodes completed:", len(all_paths))
        print("Episode lengths:", episode_lengths)
        print("Total deaths:", agent.deaths)
        print("Total replans:", agent.replans)
        print("Unique cells explored:", len(agent.visited))

        final_path = all_paths[-1] if all_paths else []
        render(env, path=final_path, start=start, goal=goal,
               background_image=file, save="maze_navigation.png", agent=agent)

        evaluator = Evaluator(file, model, classes, grid_size)
        metrics = evaluator.evaluate_agent(5)
        print('\nEvaluation Metrics:')
        for k, v in metrics.items():
            print(f'{k}: {v}')
