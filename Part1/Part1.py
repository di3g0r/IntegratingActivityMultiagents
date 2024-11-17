import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import random
from typing import List, Tuple, Dict
import time
from IPython.display import clear_output
from owlready2 import *
from ultralytics import YOLO #Part 3 Comp. Vision

# Open and load the warehouse ontology
onto = get_ontology("warehouse_ontology.owl")
onto.load()

with onto:
    # Define classes
    class Robot(Thing):
        pass

    class Warehouse(Thing):
        pass

    class Direction(Thing):
        pass

    class Cell(Thing):
        pass

    class Stack(Cell):
        pass

    class Wall(Cell):
        pass

    class EmptyCell(Cell):
        pass

    class ObjectCell(Cell):
        pass

    # Define object properties
    class hasRobot(ObjectProperty):
        domain = [Warehouse]
        range = [Robot]

    class hasDirection(ObjectProperty):
        domain = [Robot]
        range = [Direction]

    class hasCell(ObjectProperty):
        domain = [Warehouse]
        range = [Cell]

    class hasStack(ObjectProperty):
        domain = [Warehouse]
        range = [Stack]

    # Define data properties
    class hasXPosition(DataProperty):
        domain = [Robot]
        range = [int]

    class hasYPosition(DataProperty):
        domain = [Robot]
        range = [int]

    class hasId(DataProperty):
        domain = [Robot]
        range = [int]

    class isCarryingObject(DataProperty):
        domain = [Robot]
        range = [bool]

    class hasMovements(DataProperty):
        domain = [Robot]
        range = [int]

    class hasWidth(DataProperty):
        domain = [Warehouse]
        range = [int]

    class hasHeight(DataProperty):
        domain = [Warehouse]
        range = [int]

    class hasStackHeight(DataProperty):
        domain = [Stack]
        range = [int]

# Save the updated ontology
onto.save(file="warehouse_ontology.owl")

class CellType(Enum):
    EMPTY = 0
    WALL = 1
    OBJECT = 2
    ROBOT_EMPTY = 3
    ROBOT_CARRYING = 4
    STACK = 5

class Direction(Enum):
    NORTH = (0, -1)
    SOUTH = (0, 1)
    EAST = (1, 0)
    WEST = (-1, 0)

    @classmethod
    def random_direction(cls):
        return random.choice(list(cls))
    
class Robot:
    def __init__(self, x: int, y: int, id: int):
        self.x = x
        self.y = y
        self.id = id
        self.direction = random.choice(list(Direction))
        self.carrying_object = False
        self.movements = 0
        self.previous_direction = None
        self.stuck_count = 0
        self.last_positions = []
        self.max_history = 10
        self.straight_line_steps = 0
        self.min_straight_steps = 10  # direction changes
        self.random_turn_chance = 0.1
        self.vision_model = YOLO("yolov8n.pt") #Pretrained YOLO model for part 3

    def sense_environment(self, warehouse) -> dict:
      """Returns a dictionary with information about surrounding cells"""
      surroundings = {}

      # Check all four directions
      for direction in Direction:
          dx, dy = direction.value
          check_x = self.x + dx
          check_y = self.y + dy

          if warehouse._is_valid_position(check_x, check_y):
              cell_type = warehouse.grid[check_y, check_x]
              surroundings[direction] = {
                  'type': cell_type,
                  'position': (check_x, check_y),
                  'can_stack': (warehouse._can_stack_object(check_x, check_y)
                              if cell_type == CellType.STACK.value
                              else False)
              }

      return surroundings

    def update_position_history(self):
        """Update history of positions"""
        self.last_positions.append((self.x, self.y))
        if len(self.last_positions) > self.max_history:
            self.last_positions.pop(0)

    def choose_direction(self, surroundings: dict, target_position: tuple = None) -> Direction:
        """Choose new direction with strong collision avoidance"""
        # Get all available directions (not blocked)
        available_directions = [
            direction for direction, info in surroundings.items()
            if info['type'] == CellType.EMPTY.value
        ]

        # If current direction is blocked, must change
        current_blocked = (self.direction in surroundings and
                         surroundings[self.direction]['type'] != CellType.EMPTY.value)

        if current_blocked:
            if available_directions:
                # Choose a new random available direction
                return random.choice(available_directions)
            else:
                # If no available directions, rotate 180 degrees
                dx, dy = self.direction.value
                return Direction((-dx, -dy))

        # If we have a target and aren't blocked
        if target_position and not current_blocked:
            tx, ty = target_position
            dx = tx - self.x
            dy = ty - self.y

            # Try to move towards target
            if abs(dx) > abs(dy):
                preferred = Direction.EAST if dx > 0 else Direction.WEST
            else:
                preferred = Direction.SOUTH if dy > 0 else Direction.NORTH

            # Only move towards target if that direction is clear
            if (preferred in surroundings and
                surroundings[preferred]['type'] == CellType.EMPTY.value):
                return preferred

        # Random direction change chance or if we've gone straight too long
        if (random.random() < self.random_turn_chance or
            self.straight_line_steps >= self.min_straight_steps):
            if available_directions:
                return random.choice(available_directions)

        # If no reason to change and current direction is clear, keep going
        if not current_blocked:
            return self.direction

        # Default to random available direction
        return random.choice(available_directions) if available_directions else self.direction

    def move_forward(self):
        """Move robot forward in current direction"""
        self.previous_direction = self.direction
        dx, dy = self.direction.value
        self.x += dx
        self.y += dy
        self.movements += 1
        self.straight_line_steps += 1
        self.stuck_count = 0
        self.update_position_history()

    def check_side_actions(self, surroundings: dict) -> tuple[bool, tuple]:
        """Check if any side actions (pickup/stack) are possible"""
        for direction, info in surroundings.items():
            if self.carrying_object:
                if (info['type'] == CellType.STACK.value and
                    info['can_stack']):
                    return True, info['position']
            else:
                if info['type'] == CellType.OBJECT.value:
                    return True, info['position']
        return False, None

    def pick_up_object(self):
        """Pick up an object"""
        self.carrying_object = True
        print(f"\033[93mRobot {self.id} picked up an object at position ({self.x}, {self.y})\033[0m")

    def drop_object(self):
        """Drop carried object"""
        self.carrying_object = False

    # Part 3 functions
    def capture_camera_feed(self, warehouse):
      """Simulate capturing an image with the robot's camera."""
      fov_size = 3  # Define the size of the field of view
      dx, dy = self.direction.value
      start_x, start_y = self.x + dx, self.y + dy

      # Extract a subsection of the grid representing the FOV
      fov = warehouse.grid[max(start_y-1, 0):start_y+fov_size, max(start_x-1, 0):start_x+fov_size]

      # Convert the fov to a dummy RGB image (3 channels)
      image = np.stack([fov] * 3, axis=-1).astype(np.uint8) * 255  # Scale grid values to 0-255
      return image

    def detect_objects(self, image):
      """Process the image using the vision model."""
      # Ensure the image is a numpy array
      image = np.asarray(image)

      # Run YOLO model on the preprocessed image
      results = self.vision_model.predict(image)

      # Extract detection results
      detections = []
      for result in results:
          if result.boxes:  # Check if boxes are detected
              for box in result.boxes:
                  # Each box contains xyxy coordinates, confidence, and class
                  confidence = box.conf.item()  # Confidence score
                  cls = int(box.cls.item())    # Class ID
                  label = self.vision_model.names[cls]  # Class name from the model
                  detections.append((label, confidence))

      return detections

    def act_on_detections(self, detections):
        """Perform actions based on detected objects."""
        for obj, confidence in detections:
            if obj == "object" and not self.carrying_object:
                print(f"\033[93mRobot {self.id}: Detected {obj} with {confidence:.2f}. Picking up.\033[0m")
                self.pick_up_object()
                return True
            elif obj == "stack" and self.carrying_object:
                print(f"\033[92mRobot {self.id}: Detected stack. Dropping object.\033[0m")
                self.drop_object()
                return True
        return False
    
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

        # Place walls around the edges
        self.grid[0, :] = CellType.WALL.value
        self.grid[-1, :] = CellType.WALL.value
        self.grid[:, 0] = CellType.WALL.value
        self.grid[:, -1] = CellType.WALL.value

        self._initialize_robots(num_robots)
        self._place_objects(num_objects)

        print(f"\033[94mSimulation started with {num_robots} robots and {num_objects} objects to organize\033[0m")

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
                0 <= y < self.height and
                self.grid[y, x] != CellType.WALL.value)

    def _get_cell_in_front(self, robot: Robot) -> Tuple[int, int]:
        dx, dy = robot.direction.value
        new_x = robot.x + dx
        new_y = robot.y + dy
        return new_x, new_y

    def _get_cell_left(self, robot: Robot) -> Tuple[int, int]:
        dx, dy = robot.direction.value
        left_dx, left_dy = -dy, dx  # Rotate 90° counterclockwise
        return robot.x + left_dx, robot.y + left_dy

    def _get_cell_right(self, robot: Robot) -> Tuple[int, int]:
        dx, dy = robot.direction.value
        right_dx, right_dy = dy, -dx  # Rotate 90° clockwise
        return robot.x + right_dx, robot.y + right_dy

    def _find_nearest_stack_or_empty(self, x: int, y: int) -> Tuple[int, int]:
        min_distance = float('inf')
        best_position = None

        # First priority: Check existing non-full stacks
        for i in range(1, self.height-1):
            for j in range(1, self.width-1):
                if self.grid[i, j] == CellType.STACK.value and self._can_stack_object(j, i):
                    distance = abs(x - j) + abs(y - i)
                    if distance < min_distance:
                        min_distance = distance
                        best_position = (j, i)

        # If no suitable stack found, find the nearest empty space
        if best_position is None:
            for i in range(1, self.height-1):
                for j in range(1, self.width-1):
                    if self.grid[i, j] == CellType.EMPTY.value:
                        adjacent_to_wall = (
                            self.grid[i-1, j] == CellType.WALL.value or
                            self.grid[i+1, j] == CellType.WALL.value or
                            self.grid[i, j-1] == CellType.WALL.value or
                            self.grid[i, j+1] == CellType.WALL.value
                        )
                        adjacent_to_stack = (
                            self.grid[i-1, j] == CellType.STACK.value or
                            self.grid[i+1, j] == CellType.STACK.value or
                            self.grid[i, j-1] == CellType.STACK.value or
                            self.grid[i, j+1] == CellType.STACK.value
                        )

                        distance = abs(x - j) + abs(y - i)
                        if adjacent_to_wall or adjacent_to_stack:
                            distance -= 2

                        if distance < min_distance:
                            min_distance = distance
                            best_position = (j, i)

        return best_position

    def _can_stack_object(self, x: int, y: int) -> bool:
        return (x, y) not in self.stacks or self.stacks[(x, y)] < self.max_stack_height

    def step(self):
      self.steps += 1

      for robot in self.robots:
          # Clear current position
          self.grid[robot.y, robot.x] = CellType.EMPTY.value

          # Get surroundings and check for side actions
          surroundings = robot.sense_environment(self)

          # Vision-based object detection
          fov = robot.capture_camera_feed(self)
          detections = robot.detect_objects(fov)
          acted = robot.act_on_detections(detections)

          if not acted:
              # Existing movement logic
              can_act, action_pos = robot.check_side_actions(surroundings)

              if can_act:
                  # Perform side action if possible
                  ax, ay = action_pos
                  if robot.carrying_object and self.grid[ay, ax] == CellType.STACK.value:
                      # Add to existing stack
                      new_height = self.stacks.get((ax, ay), 0) + 1
                      self.stacks[(ax, ay)] = new_height
                      robot.drop_object()
                      print(f"\033[92mRobot {robot.id} added to stack at ({ax}, {ay}), new height: {new_height}\033[0m")

                      if new_height == self.max_stack_height:
                          print(f"\033[95m🎉 Stack at ({ax}, {ay}) is complete! Height: {new_height}\033[0m")
                  elif not robot.carrying_object and self.grid[ay, ax] == CellType.OBJECT.value:
                      # Pick up object
                      self.grid[ay, ax] = CellType.EMPTY.value
                      robot.pick_up_object()
              else:
                  # Movement logic if no side actions
                  if robot.carrying_object:
                      # Find best place to create or add to stack
                      target_x, target_y = self._find_nearest_stack_or_empty(robot.x, robot.y)

                      # Choose direction based on target
                      new_direction = robot.choose_direction(surroundings, (target_x, target_y))

                      if robot.direction != new_direction:
                          robot.direction = new_direction

                      front_x, front_y = self._get_cell_in_front(robot)

                      if front_x == target_x and front_y == target_y:
                          if self.grid[front_y, front_x] == CellType.EMPTY.value:
                              # Create new stack
                              self.grid[front_y, front_x] = CellType.STACK.value
                              self.stacks[(front_x, front_y)] = 1
                              robot.drop_object()
                              print(f"\033[92mRobot {robot.id} created new stack at ({front_x}, {front_y})\033[0m")
                      elif self.grid[front_y, front_x] == CellType.EMPTY.value:
                          robot.move_forward()
                      else:
                          robot.stuck_count += 1
                  else:
                      # Choose direction for exploration
                      new_direction = robot.choose_direction(surroundings)
                      if robot.direction != new_direction:
                          robot.direction = new_direction

                      front_x, front_y = self._get_cell_in_front(robot)

                      if self.grid[front_y, front_x] == CellType.EMPTY.value:
                          robot.move_forward()
                      else:
                          robot.stuck_count += 1

          # Update robot position on grid with appropriate state
          self.grid[robot.y, robot.x] = (
              CellType.ROBOT_CARRYING.value if robot.carrying_object
              else CellType.ROBOT_EMPTY.value
          )


    def is_organized(self) -> bool:
        # Count total objects (in stacks and being carried)
        total_stacked = sum(self.stacks.values())
        objects_carried = sum(1 for robot in self.robots if robot.carrying_object)
        remaining_objects = np.sum(self.grid == CellType.OBJECT.value)

        # All objects should be in complete stacks
        return (total_stacked + objects_carried + remaining_objects == self.initial_objects and
                remaining_objects == 0 and
                objects_carried == 0 and
                all(height == self.max_stack_height for height in self.stacks.values()))

    def visualize(self):
        plt.clf()
        # Updated colormap to include different robot states
        cmap = plt.cm.colors.ListedColormap(['white', 'black', 'red', 'blue', 'purple', 'green'])
        plt.imshow(self.grid, cmap=cmap)

        # Add arrows for robot directions
        for robot in self.robots:
            dx, dy = robot.direction.value
            color = 'orange' if robot.carrying_object else 'yellow'
            plt.arrow(robot.x, robot.y, dx*0.3, dy*0.3,
                     head_width=0.3, head_length=0.3, fc=color, ec=color)

        # Add stack heights
        for (x, y), height in self.stacks.items():
            color = 'white' if height < self.max_stack_height else 'yellow'
            plt.text(x, y, str(height), ha='center', va='center', color=color)

        plt.grid(True)
        plt.title(f'Step: {self.steps}')
        plt.pause(0.1)

def run_simulation(width: int, height: int, num_robots: int, num_objects: int,
                  max_steps: int = 1000):
    warehouse = Warehouse(width, height, num_robots, num_objects)
    plt.figure(figsize=(10, 10))

    try:
        while not warehouse.is_organized() and warehouse.steps < max_steps:
            warehouse.step()
            warehouse.visualize()

            # Print progress every 100 steps
            if warehouse.steps % 100 == 0:
                total_stacked = sum(warehouse.stacks.values())
                objects_carried = sum(1 for robot in warehouse.robots if robot.carrying_object)
                remaining = np.sum(warehouse.grid == CellType.OBJECT.value)
                print(f"\nStep {warehouse.steps}:")
                print(f"Objects in stacks: {total_stacked}")
                print(f"Objects carried: {objects_carried}")
                print(f"Objects remaining: {remaining}")

    except Exception as e:
        print(f"Error during simulation: {str(e)}")
        import traceback
        traceback.print_exc()

    print(f"\n\033[94mSimulation completed in {warehouse.steps} steps\033[0m")
    for robot in warehouse.robots:
        print(f"\033[96mRobot {robot.id} made {robot.movements} movements\033[0m")

    # Final status
    total_stacked = sum(warehouse.stacks.values())
    objects_carried = sum(1 for robot in warehouse.robots if robot.carrying_object)
    remaining = np.sum(warehouse.grid == CellType.OBJECT.value)
    print(f"\nFinal status:")
    print(f"Objects in stacks: {total_stacked}")
    print(f"Objects carried: {objects_carried}")
    print(f"Objects remaining: {remaining}")

    return warehouse

warehouse = run_simulation(
    width=15,
    height=15,
    num_robots=5,
    num_objects=20
)