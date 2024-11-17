from flask import Flask, jsonify, request
import numpy as np
import random
from enum import Enum
from typing import List, Tuple, Dict
import logging
from datetime import datetime

app = Flask(__name__)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Direction(Enum):
    NORTH = (0, -1)
    SOUTH = (0, 1)
    EAST = (1, 0)
    WEST = (-1, 0)

    @classmethod
    def random_direction(cls):
        return random.choice(list(cls))

class CellType(Enum):
    EMPTY = 0
    WALL = 1
    OBJECT = 2
    ROBOT_EMPTY = 3
    ROBOT_CARRYING = 4
    STACK = 5

class Robot:
    def __init__(self, x: int, y: int, id: int):
        self.x = x
        self.y = y
        self.id = id
        self.direction = random.choice(list(Direction))
        self.carrying_object = False
        self.movements = 0
        self.objects_stacked = 0
        self.stuck_count = 0
        self.last_positions = []
        self.max_history = 10
        self.state = 'SEARCHING'  # States: SEARCHING, GOING_TO_STACK
        self.target_stack = None
        self.avoidance_cooldown = 0

    def sense_environment(self, warehouse) -> dict:
        surroundings = {}
        for direction in Direction:
            dx, dy = direction.value
            check_x = self.x + dx
            check_y = self.y + dy
            if warehouse._is_valid_position(check_x, check_y):
                cell_type = warehouse.grid[check_y, check_x]
                surroundings[direction] = {
                    'type': cell_type,
                    'position': (check_x, check_y)
                }
        return surroundings

    def is_near_wall(self, warehouse) -> bool:
        surroundings = self.sense_environment(warehouse)
        return any(info['type'] == CellType.WALL.value for info in surroundings.values())

    def choose_direction(self, surroundings: dict, target_pos: tuple = None) -> Direction:
        if self.avoidance_cooldown > 0:
            self.avoidance_cooldown -= 1
            available_directions = [
                direction for direction, info in surroundings.items()
                if info['type'] == CellType.EMPTY.value
            ]
            if available_directions:
                return random.choice(available_directions)
            return self.direction

        if target_pos and self.carrying_object:
            tx, ty = target_pos
            dx = tx - self.x
            dy = ty - self.y
            
            if abs(dx) > abs(dy):
                primary = Direction.EAST if dx > 0 else Direction.WEST
                secondary = Direction.SOUTH if dy > 0 else Direction.NORTH
            else:
                primary = Direction.SOUTH if dy > 0 else Direction.NORTH
                secondary = Direction.EAST if dx > 0 else Direction.WEST

            if primary in surroundings and surroundings[primary]['type'] == CellType.EMPTY.value:
                return primary
            if secondary in surroundings and surroundings[secondary]['type'] == CellType.EMPTY.value:
                return secondary

        available_directions = [
            direction for direction, info in surroundings.items()
            if info['type'] == CellType.EMPTY.value
        ]
        
        if not available_directions:
            self.avoidance_cooldown = 3
            return Direction.random_direction()
            
        return random.choice(available_directions)

    def move(self, new_x: int, new_y: int):
        self.x = new_x
        self.y = new_y
        self.movements += 1
        self.last_positions.append((new_x, new_y))
        if len(self.last_positions) > self.max_history:
            self.last_positions.pop(0)

class Warehouse:
    def __init__(self, width: int, height: int, num_robots: int, num_objects: int):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=int)
        self.robots: List[Robot] = []
        self.stacks: Dict[Tuple[int, int], int] = {}
        self.steps = 0
        self.max_stack_height = 5
        self.initial_objects = num_objects
        self.objects_stacked = 0
        self.simulation_complete = False

        # Place walls
        self.grid[0, :] = CellType.WALL.value
        self.grid[-1, :] = CellType.WALL.value
        self.grid[:, 0] = CellType.WALL.value
        self.grid[:, -1] = CellType.WALL.value

        logger.info("=== Simulation Started ===")
        logger.info(f"Initial objects: {num_objects}")
        logger.info(f"Number of robots: {num_robots}")

        self._initialize_robots(num_robots)
        self._place_objects(num_objects)

    def _log_event(self, message: str):
        logger.info(message)
        self.simulation_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {message}")

    def _initialize_robots(self, num_robots: int):
        for i in range(num_robots):
            while True:
                x = random.randint(1, self.width-2)
                y = random.randint(1, self.height-2)
                if self.grid[y, x] == CellType.EMPTY.value:
                    robot = Robot(x, y, i)
                    self.robots.append(robot)
                    self.grid[y, x] = CellType.ROBOT_EMPTY.value
                    break

    def _place_objects(self, num_objects: int):
        objects_placed = 0
        while objects_placed < num_objects:
            x = random.randint(1, self.width-2)
            y = random.randint(1, self.height-2)
            if self.grid[y, x] == CellType.EMPTY.value:
                self.grid[y, x] = CellType.OBJECT.value
                objects_placed += 1

    def _is_valid_position(self, x: int, y: int) -> bool:
        return (0 <= x < self.width and
                0 <= y < self.height)

    def _find_stack_location(self, robot: Robot) -> Tuple[int, int]:
        if robot.target_stack and self._can_stack_object(*robot.target_stack):
            return robot.target_stack

        best_distance = float('inf')
        best_location = None

        for (sx, sy), height in self.stacks.items():
            if height < self.max_stack_height:
                dist = abs(robot.x - sx) + abs(robot.y - sy)
                if dist < best_distance:
                    best_distance = dist
                    best_location = (sx, sy)

        if not best_location:
            for y in range(1, self.height-1):
                for x in range(1, self.width-1):
                    if self.grid[y, x] == CellType.EMPTY.value and self._is_near_wall(x, y):
                        dist = abs(robot.x - x) + abs(robot.y - y)
                        if dist < best_distance:
                            best_distance = dist
                            best_location = (x, y)

        return best_location

    def _is_near_wall(self, x: int, y: int) -> bool:
        return (self.grid[y-1, x] == CellType.WALL.value or
                self.grid[y+1, x] == CellType.WALL.value or
                self.grid[y, x-1] == CellType.WALL.value or
                self.grid[y, x+1] == CellType.WALL.value)

    def _can_stack_object(self, x: int, y: int) -> bool:
        return (x, y) not in self.stacks or self.stacks[(x, y)] < self.max_stack_height

    def step(self):
        self.steps += 1
        
        for robot in self.robots:
            old_x, old_y = robot.x, robot.y
            self.grid[old_y, old_x] = CellType.EMPTY.value
            
            surroundings = robot.sense_environment(self)
            
            if robot.carrying_object:
                target_stack = self._find_stack_location(robot)
                robot.target_stack = target_stack
                
                if target_stack:
                    tx, ty = target_stack
                    if (abs(robot.x - tx) <= 1 and abs(robot.y - ty) <= 1):
                        if self.grid[ty, tx] == CellType.STACK.value:
                            self.stacks[(tx, ty)] = self.stacks.get((tx, ty), 0) + 1
                        else:
                            self.grid[ty, tx] = CellType.STACK.value
                            self.stacks[(tx, ty)] = 1
                        
                        robot.carrying_object = False
                        robot.target_stack = None
                        robot.state = 'SEARCHING'
                        robot.objects_stacked += 1
                        self.objects_stacked += 1
                        logger.info(f"Robot {robot.id} dropped object at stack ({tx}, {ty}). Stack height: {self.stacks[(tx, ty)]}")
                    else:
                        new_direction = robot.choose_direction(surroundings, target_stack)
                        robot.direction = new_direction
                        dx, dy = robot.direction.value
                        new_x, new_y = robot.x + dx, robot.y + dy
                        
                        if (self._is_valid_position(new_x, new_y) and 
                            self.grid[new_y, new_x] == CellType.EMPTY.value):
                            robot.move(new_x, new_y)
            else:
                object_found = False
                for direction, info in surroundings.items():
                    if info['type'] == CellType.OBJECT.value:
                        ox, oy = info['position']
                        self.grid[oy, ox] = CellType.EMPTY.value
                        robot.carrying_object = True
                        robot.state = 'GOING_TO_STACK'
                        object_found = True
                        logger.info(f"Robot {robot.id} picked up object at ({ox}, {oy})")
                        break
                
                if not object_found:
                    new_direction = robot.choose_direction(surroundings)
                    robot.direction = new_direction
                    dx, dy = robot.direction.value
                    new_x, new_y = robot.x + dx, robot.y + dy
                    
                    if (self._is_valid_position(new_x, new_y) and 
                        self.grid[new_y, new_x] == CellType.EMPTY.value):
                        robot.move(new_x, new_y)
            
            self.grid[robot.y, robot.x] = (
                CellType.ROBOT_CARRYING.value if robot.carrying_object
                else CellType.ROBOT_EMPTY.value
            )

        # Check if simulation is complete
        if self.objects_stacked == self.initial_objects and not self.simulation_complete:
            self.simulation_complete = True
            self._log_completion_report()

    def _log_completion_report(self):
        logger.info("\n=== Simulation Complete ===")
        logger.info(f"Total steps: {self.steps}")
        logger.info("\nRobot Statistics:")
        for robot in self.robots:
            logger.info(f"Robot {robot.id}:")
            logger.info(f"  - Total movements: {robot.movements}")
            logger.info(f"  - Objects stacked: {robot.objects_stacked}")
        logger.info("\nStack Locations:")
        for (x, y), height in self.stacks.items():
            logger.info(f"Stack at ({x}, {y}): {height} objects")
        logger.info("========================")

warehouse = None

@app.route('/initialize', methods=['POST'])
def initialize_simulation():
    global warehouse
    data = request.get_json()
    width = data.get('width', 15)
    height = data.get('height', 15)
    num_robots = data.get('num_robots', 5)
    num_objects = data.get('num_objects', 20)
    warehouse = Warehouse(width, height, num_robots, num_objects)
    return jsonify({"message": "Simulation initialized"}), 200

@app.route('/step', methods=['POST'])
def step_simulation():
    global warehouse
    if warehouse:
        warehouse.step()
        return jsonify({"message": "Step executed"}), 200
    return jsonify({"error": "Simulation not initialized"}), 400

@app.route('/simulation-state', methods=['GET'])
def get_simulation_state():
    if warehouse:
        state = {
            "grid": warehouse.grid.tolist(),
            "robots": [
                {
                    "id": robot.id,
                    "x": robot.x,
                    "y": robot.y,
                    "direction": robot.direction.name,
                    "carrying_object": robot.carrying_object,
                    "state": robot.state
                }
                for robot in warehouse.robots
            ],
            "stacks": [
                {
                    "x": x,
                    "y": y,
                    "height": height
                }
                for (x, y), height in warehouse.stacks.items()
            ]
        }
        return jsonify(state), 200
    return jsonify({"error": "Simulation not initialized"}), 400

if __name__ == "__main__":
    app.run(debug=True)