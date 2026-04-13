from enum import Enum
from typing import List, Tuple
from PIL import Image, ImageDraw
import heapq

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



# loading Image → Maze
def ImageToGrid(pixel):
    r, g, b = pixel
    return 1 if (r < 60 and g < 60 and b < 60) else 0


def image_to_maze(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((1026, 1026), Image.NEAREST)

    w, h = img.size
    maze = []

    for y in range(h):
        row = []
        for x in range(w):
            row.append(ImageToGrid(img.getpixel((x, y))))
        maze.append(row)

    start = None
    goal = None

    for x in range(w):
        if maze[h - 1][x] == 0:
            start = (h - 1, x)
            break

    for x in range(w):
        if maze[0][x] == 0:
            goal = (0, x)
            break

    return maze, start, goal



# Maze enviroment
class MazeEnvironment:
    def __init__(self, image_path):
        self.maze, self.start, self.goal = image_to_maze(image_path)
        self.reset()

    def reset(self):
        self.pos = self.start
        self.path_history = [self.pos]
        
        self.turns_taken = 0
        self.deaths = 0
        self.confused = 0
        self.cells_explored = set([self.pos])

        return self.pos

    def step(self, actions: List[Action]) -> TurnResult:
        result = TurnResult()

        for action in actions:
            y, x = self.pos

            if action == Action.MOVE_UP:
                ny, nx = y - 1, x
            elif action == Action.MOVE_DOWN:
                ny, nx = y + 1, x
            elif action == Action.MOVE_LEFT:
                ny, nx = y, x - 1
            elif action == Action.MOVE_RIGHT:
                ny, nx = y, x + 1
            else:
                ny, nx = y, x

            if not (0 <= ny < len(self.maze) and 0 <= nx < len(self.maze[0])):
                result.wall_hits += 1
                continue

            if self.maze[ny][nx] == 1:
                result.wall_hits += 1
                continue

            self.pos = (ny, nx)
            self.path_history.append(self.pos)
            self.cells_explored.add(self.pos)

            result.actions_executed += 1

            if self.pos == self.goal:
                result.is_goal_reached = True
                break

        result.current_position = self.pos
        self.turns_taken += 1

        return result

   
    def get_episode_stats(self) -> dict:
        return {
            "turns_taken": self.turns_taken,
            "deaths": self.deaths,
            "confused": self.confused,
            "cells_explored": len(self.cells_explored),
            "goal_reached": self.pos == self.goal,
        }

# A* agent
class Agent:
    def __init__(self):
        self.maze = None
        self.start = None
        self.goal = None
        self.path = []
        self.index = 0

    def init_astar(self):
        def h(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        open_list = [(0, self.start)]
        parent = {self.start: None}
        g = {self.start: 0}

        while open_list:
            _, cur = heapq.heappop(open_list)

            if cur == self.goal:
                break

            y, x = cur

            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                ny, nx = y + dy, x + dx

                if not (0 <= ny < len(self.maze) and 0 <= nx < len(self.maze[0])):
                    continue
                if self.maze[ny][nx] == 1:
                    continue

                nxt = (ny, nx)
                cost = g[cur] + 1

                if nxt not in g or cost < g[nxt]:
                    g[nxt] = cost
                    heapq.heappush(open_list, (cost + h(nxt, self.goal), nxt))
                    parent[nxt] = cur

        path = []
        cur = self.goal

        if cur in parent:
            while cur:
                path.append(cur)
                cur = parent[cur]
            path.reverse()

        self.path = path

    def plan_turn(self, last_result: TurnResult) -> List[Action]:
        actions = []

        for _ in range(5):
            if self.index >= len(self.path) - 1:
                break

            y1, x1 = self.path[self.index]
            y2, x2 = self.path[self.index + 1]
            self.index += 1

            if y2 < y1:
                actions.append(Action.MOVE_UP)
            elif y2 > y1:
                actions.append(Action.MOVE_DOWN)
            elif x2 < x1:
                actions.append(Action.MOVE_LEFT)
            else:
                actions.append(Action.MOVE_RIGHT)

        return actions if actions else [Action.WAIT]

    def reset_episode(self):
        self.index = 0
        

# Visualizing the Maze path
def visualizer(maze, path_history, optimal_path, output="Maze_navigation.png"):
    img = Image.new("RGB", (len(maze[0]), len(maze)), "white")
    draw = ImageDraw.Draw(img)

    for y in range(len(maze)):
        for x in range(len(maze[0])):
            if maze[y][x] == 1:
                draw.point((x, y), fill="black")

    for y, x in path_history:
        draw.point((x, y), fill="green")

    if optimal_path:
        for i in range(len(optimal_path) - 1):
            y1, x1 = optimal_path[i]
            y2, x2 = optimal_path[i + 1]
            draw.line((x1, y1, x2, y2), fill="red", width=1)

    img.save(output) 


# Evaluator
class Evaluator: 

    def __init__(self, maze_path):
        self.maze_path = maze_path

    def evaluate_agent(self, agent: Agent, maze_id: str, num_episodes: int = 5) -> dict:
        success = 0
        total_turns = 0
        total_path_length = 0
        total_unique = 0
        total_steps = 0
        total_deaths = 0  

        for _ in range(num_episodes):
            env = MazeEnvironment(self.maze_path)

            agent.maze = env.maze
            agent.start = env.start
            agent.goal = env.goal
            agent.init_astar()
            agent.reset_episode()

            last = None
            steps = 0

            while steps < 20000:
                actions = agent.plan_turn(last)
                last = env.step(actions)

                if last.is_goal_reached:
                    success += 1
                    break

                steps += 1

            total_turns += steps
            
            path_len = len(env.path_history)
            unique_cells = len(set(env.path_history))

            total_path_length += path_len
            total_unique += unique_cells
            total_steps += path_len
            total_deaths += env.deaths  

        self.metrics = {
            "success_rate": success / num_episodes,
            "avg_turns": total_turns / num_episodes,
            "death_rate": total_deaths / (total_turns + 1),  
            "avg_path_length": total_path_length / num_episodes,
            "exploration_efficiency": (
                total_unique / total_steps if total_steps > 0 else 0
            )
        }

        return self.metrics

    def get_metrics(self) -> dict:
        return self.metrics


# main
if __name__ == "__main__":

    maze_file = input("Enter maze image filename: ").strip()
    env = MazeEnvironment(maze_file)

    agent = Agent()

    agent.maze = env.maze
    agent.start = env.start
    agent.goal = env.goal
    agent.init_astar()

    pos = env.reset()

    last = None
    steps = 0

    while steps < 10000:
        actions = agent.plan_turn(last)
        last = env.step(actions)

        if last.is_goal_reached:            
            break

        steps += 1

    visualizer(env.maze, env.path_history, agent.path)

    evaluator = Evaluator(maze_file)
    print(evaluator.evaluate_agent(agent, "training", 3))
