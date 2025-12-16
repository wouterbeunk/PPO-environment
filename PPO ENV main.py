import os
os.environ["DISABLE_TENSORBOARD"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PPO_data_loader import data_loader
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import sys
import csv
from datetime import datetime
from stable_baselines3.common.vec_env import VecNormalize
from torch.utils.tensorboard import SummaryWriter
import random
import itertools

# This class uses base callback from SB3 to write the results during each training step
class EnhancedLogging(BaseCallback):
    """
    Enhanced callback for logging training progress to both CSV and TensorBoard.
    Tracks material handling costs, rewards, and learning rates.
    """

    def __init__(self, csv_filename="ppo_training_log.csv",
                 tensorboard_log="./tensorboard_logs/", verbose=0):
        super().__init__(verbose)
        self.csv_filename = csv_filename
        self.tensorboard_log = tensorboard_log
        self.header_written = False
        self.step_count = 0

        # Create tensorboard logs directory
        os.makedirs(tensorboard_log, exist_ok=True)
        self.writer = SummaryWriter(log_dir=tensorboard_log)

        # Create CSV file with headers
        print(f"Logging to CSV: {csv_filename}")
        print(f"Logging to TensorBoard: {tensorboard_log}")
        print(f"View with: tensorboard --logdir={tensorboard_log}")

        with open(self.csv_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Timestep",
                "Cost",
                "Reward",
                "Episode",
                "Learning_Rate",
                "Timestamp"
            ])

    def _on_step(self) -> bool:
        self.step_count += 1
        infos = self.locals.get('infos', [])
        rewards = self.locals.get('rewards', None)

        # Process rewards
        if rewards is None:
            rewards_list = []
        else:
            try:
                rewards_list = list(rewards)
            except Exception:
                rewards_list = [float(rewards)]

        # Log each environment's data
        for env_idx, info in enumerate(infos):
            cost = None
            reward = None

            if isinstance(info, dict):
                cost = info.get('cost', None)

            if env_idx < len(rewards_list):
                try:
                    reward = float(rewards_list[env_idx])
                except Exception:
                    reward = None

            # Scale costs if needed
            if cost is not None:
                cost = cost / 10000.0

            # Write to CSV
            if cost is not None or reward is not None:
                with open(self.csv_filename, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        self.num_timesteps,
                        cost if cost is not None else "",
                        reward if reward is not None else "",
                        self.num_timesteps // 1000,
                        self.model.learning_rate,
                        datetime.now().isoformat()
                    ])

            # Write to TensorBoard
            if cost is not None:
                self.writer.add_scalar('environment/cost', cost, self.num_timesteps)
            if reward is not None:
                self.writer.add_scalar('environment/reward', reward, self.num_timesteps)

            self.writer.add_scalar('training/learning_rate',
                                   self.model.learning_rate,
                                   self.num_timesteps)

        # Flush periodically
        if self.step_count % 100 == 0:
            self.writer.flush()

        return True

    def _on_training_end(self):
        self.writer.flush()
        self.writer.close()



#Checks for any immediate errors
print("[DEBUG] Script starting...")
sys.stdout.flush()

# PPO Learning environment custom to Ekro's by-products facility
class ContinuousIrregularFLPEnv(gym.Env):
    # two render_modes are set: human if you want to visualize it, rgb_array for computational processing
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self,
                 n_facilities,
                 max_steps: int = 200,
                 render_mode = None,
                 seed: int = None,
                 excel_data=None,
                 fixed_scenario=None):

        super().__init__()

        np.random.seed(seed)

        self.max_steps = max_steps
        self.render_mode = render_mode

        # Defining main placement areas (x, y, length, width (x and y coordinates are bottom left corner))
        self.main_area = (0, 0, 29.0, 28.82)
        self.top_right_area = (12.04, 29.1, 17.06, 10.88)
        self.mid_right_area = (29.1, 18.5, 7.9, 4.2)

        # total area of the entire field (including infeasible areas)
        self.total_length = 37.0
        self.total_width = 39.98

        # setting the restricted zones (x, y, length, width (x and y coordinates are bottom left corner))
        self.restricted_zones = [
            (0, 4.75, 0.5, 0.5),
            (6.18, 4.75, 0.5, 0.5),
            (14.18, 4.75, 0.5, 0.5),
            (22.18, 4.75, 0.5, 0.5),
            (28.36, 4.75, 0.5, 0.5),
            (0, 10.75, 0.5, 0.5),
            (6.18, 10.75, 0.5, 0.5),
            (14.18, 10.75, 0.5, 0.5),
            (22.18, 10.75, 0.5, 0.5),
            (28.36, 10.75, 0.5, 0.5),
            (0, 16.75, 0.5, 0.5),
            (6.18, 16.75, 0.5, 0.5),
            (14.18, 16.75, 0.5, 0.5),
            (22.18, 16.75, 0.5, 0.5),
            (28.36, 16.75, 0.5, 0.5),
            (0, 22.75, 0.5, 0.5),
            (6.18, 22.75, 0.5, 0.5),
            (14.18, 22.75, 0.5, 0.5),
            (22.18, 22.75, 0.5, 0.5),
            (28.36, 22.75, 0.5, 0.5),
            (0, 28.52, 0.5, 0.5),
            (6.18, 28.52, 0.5, 0.5),
            (14.18, 28.52, 0.5, 0.5),
            (22.18, 28.52, 0.5, 0.5),
            (28.36, 28.52, 0.5, 0.5),
            (14.04, 29.44, 0.5, 0.5),
            (19.01, 29.44, 0.5, 0.5),
            (24.01, 29.44, 0.5, 0.5),
            (19.01, 34.44, 0.5, 0.5),
            (19.01, 39.44, 0.5, 0.5),
            (6.49, 17.38, 2.3, 4.36),
            (0, 17.25, 6.38, 5.51),
            (29.1, 18.5, 7.9, 4.2),
            (29.0, 0.0, 11, 18.5),
            (37, 18.5, 3, 4.2),
            (29.1, 22.7, 10.9, 17.3),
            (0.0, 29.1, 12.04, 10.9)
        ]

        # Loading the excel data
        if excel_data is not None:
            extracted_data = excel_data
        else:
            filepath = "C:\\Users\\beunk\\Downloads\\Bij-producten meta data.xlsx"
            print(f"[DEBUG] Loading Excel from: {filepath}")
            sys.stdout.flush()
            data_import = data_loader(filepath)
            extracted_data = data_import.excel_file_loading()
            print("[DEBUG] Excel loaded successfully in __init__")
            sys.stdout.flush()

        # Extracting the data from the Excel file
        stations_df = extracted_data["Stations"]
        high_flow_df = extracted_data["High flow matrix"]
        regular_flow_df = extracted_data["Regular flow matrix"]
        low_flow_df = extracted_data["Low flow matrix"]
        entry_points_df = extracted_data["Entry Points"]

        # Creating variables to link the machines of both sheets with each other
        station_refs = [s for s in stations_df["Number"] if str(s).startswith("S")]
        entry_refs = [idx for idx in regular_flow_df.index if str(idx).startswith("E")]

        # Incase there are non values in the flow matrices, then they are filled with 0
        def clean_flow(df):
            return df.loc[
                [r for r in df.index if str(r).startswith(("S", "E"))],
                [c for c in df.columns if str(c).startswith("S")]
            ].fillna(0).astype(float)

        # Set the flow matrices
        flow_high_df = clean_flow(high_flow_df)
        flow_regular_df = clean_flow(regular_flow_df)
        flow_low_df = clean_flow(low_flow_df)

        # Store the flow scenarios as a dictionary with according weights
        self.flow_scenarios = {
            "high": {"matrix": flow_high_df, "weight": 1 / 12},
            "regular": {"matrix": flow_regular_df, "weight": 9 / 12},
            "low": {"matrix": flow_low_df, "weight": 2 / 12}, }

        # Set default flow matrix to regular
        self.flow_matrix = flow_regular_df
        self.fixed_scenario = fixed_scenario

        # Store the station and entry references of the excel sheets: E.g. S0 and E0
        self.station_references = station_refs
        self.entry_references = entry_refs

        # Set the dimensions of machines and how many machines there are
        self.machine_dimensions = stations_df[["Length", "Width"]].values.tolist()
        self.n_facilities = len(station_refs)

        # Setting the product flow entry point coordinates from the excel file
        self.entry_points_coordinates = {}
        for _, row in entry_points_df.iterrows():
            entry_point = row["EntryPoint"]
            x = float(row["X"])
            y = float(row["Y"])

            # format the entry point dictionary to Station: {entry point, flow}
            self.entry_points_coordinates[entry_point] = (x, y)

        # Setting the system exit points: when by-products leave the department to storage/customer (two doors they could leave through)
        self.exit_points = {
            "Exit1": {"x": 0.0, "y": 12.5},
            "Exit2": {"x": 0.0, "y": 26.4},
        }

        # S8-S11 were separated into parts of the packaging machines. Creating an additional parameter for close placement
        self.connected_line = [8, 9, 10, 11]
        self.connected_line_spacing = 0.0

        # creating the action space dictionary, because facilityID is discrete and placement should be continuous
        # both x and y coordinates of the facilities can be moved in a continuous action space ranging from -1 until 1
        # agent can choose whether to not rotate = 0 or rotate = 1
        # Flatten action space: [dx, dy, rotate]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1], dtype=np.float32),
            shape=(3,),
            dtype=np.float32
        )
        # Track which facility to modify in sequence
        self.current_facility_idx = 0

        # setting the total possible area where the agent can look (so from 0, 0 until the max length and width)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_facilities * 2 + self.n_facilities,),
            dtype=np.float32
        )

        # Initializing variables
        self.current_layout = None
        self.step_count = 0
        self.previous_cost = float('inf')
        # random number generation based on the seed
        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)

        self.debug_mode = False

    # function selects the flow scenario based on the weights of the flow matrices
    def _select_random_flow_scenario(self):
        scenario_names = list(self.flow_scenarios.keys())
        scenario_weights = [self.flow_scenarios[name]["weight"] for name in scenario_names]
        chosen_name = self.rng.choice(scenario_names, p=np.array(scenario_weights) / np.sum(scenario_weights))
        chosen_matrix = self.flow_scenarios[chosen_name]["matrix"]
        self.flow_matrix = chosen_matrix
        self.current_scenario = chosen_name

    # check whether facilities can be placed within the separate areas, so they will not be placed outside
    def _is_in_irregular_area(self, x, y, l, w):
        def inside_main(px, py):
            return 0 <= px <= 29.0 and 0 <= py <= 28.82

        def inside_top_right(px, py):
            return 12.04 <= px <= 29.1 and 29.1 <= py <= 39.98

        def inside_mid_right(px, py):
            return 29.1 <= px <= 37 and 18.5 <= py <= 22.7

        # if the corners are placed outside the areas, they return False
        corners = [(x, y), (x + l, y), (x, y + w), (x + l, y + w)]
        for (px, py) in corners:
            if not (inside_main(px, py) or inside_top_right(px, py) or inside_mid_right(px, py)):
                return False
        return True

    # creating a function for restricted zones
    # x+l is cannot overlap with the x of restricted zone for example
    def _is_in_restricted_zone(self, x, y, l, w):
        for idx, (rx, ry, rl, rw) in enumerate(self.restricted_zones):
            overlap = not (x + l <= rx or rx + rl <= x or y + w <= ry or ry + rw <= y)
            if overlap:
                if self.debug_mode:
                    print(f"[DEBUG] Facility overlaps restricted zone {idx}: "
                          f"Facility=({x:.2f},{y:.2f},{l:.2f},{w:.2f}), "
                          f"Restricted=({rx:.2f},{ry:.2f},{rl:.2f},{rw:.2f})")
                return True
        return False

    # combining previous function and determining whether the placement is valid or not
    def _is_valid_placement(self, x, y, length, width, facility_id, layout=None):
        if not self._is_in_irregular_area(x, y, length, width):
            if self.debug_mode:
                print(f"[DEBUG] Facility {facility_id} placed outside allowed area.")
            return False

        if self._is_in_restricted_zone(x, y, length, width):
            if self.debug_mode:
                print(f"[DEBUG] Facility {facility_id} placed inside restricted zone.")
            return False

        if layout is None:
            layout = self.current_layout

        for other_id, (ox, oy, orig_ol, orig_ow, o_orientation) in layout.items():
            if other_id == facility_id:
                continue

            # compute actual dims of the other facility
            if o_orientation == 1:
                other_l, other_w = orig_ow, orig_ol
            else:
                other_l, other_w = orig_ol, orig_ow

            overlap = not (x + length <= ox or ox + other_l <= x or
                           y + width <= oy or oy + other_w <= y)
            if overlap:
                if self.debug_mode:
                    print(f"[DEBUG] Facility {facility_id} overlaps machine {other_id}.")
                    print(f"        F: ({x:.2f},{y:.2f},{length:.2f},{width:.2f}) "
                          f"O: ({ox:.2f},{oy:.2f},{other_l:.2f},{other_w:.2f})")
                return False

        return True

    # This function calculates the distances of all machines on average to restricted zones
    # Needs to be as low as possible
    def _calculate_proximity_to_restricted_areas(self):
        rz = np.array(self.restricted_zones)  # Array of the restricted zones
        total_distance = 0.0 # Variable for summing the total distances
        for fid, (x, y, length, width, orientation) in self.current_layout.items(): # loop over the workstations
            # Compute display dimensions. When the workstation is turned (orientation == 1), the width and length are switched
            dl, dw = (width, length) if orientation == 1 else (length, width)
            cx, cy = x + dl / 2.0, y + dw / 2.0 # Calculating the centres of the placed facilities

            # distances to each restricted zone
            closest_xs = np.clip(cx, rz[:, 0], rz[:, 0] + rz[:, 2])
            closest_ys = np.clip(cy, rz[:, 1], rz[:, 1] + rz[:, 3])
            dists = np.sqrt((cx - closest_xs) ** 2 + (cy - closest_ys) ** 2)
            min_dist = np.min(dists)
            total_distance += float(min_dist)
        # when there are no machines placed, infinity will be given to cause an error
        if len(self.current_layout) == 0:
            return float('inf')
        avg_distance = total_distance / len(self.current_layout)
        return avg_distance

    # function creates everytime a random layout for the learning agent
    def _generate_random_layout(self):
        layout = {}
        max_retries = 1000

        # Connected packaging line constraint
        # Place connected line S8-S11
        max_initial_spacing = 2.0  # preferred distance
        max_expand_spacing = 6.0  # maximum allowed distance if initial fails
        prev_fid = self.connected_line[0]

        # Place first machine freely
        l, w = self.machine_dimensions[prev_fid]
        # Place first machine freely, but check for validity
        placed = False
        attempts = 0
        while not placed and attempts < 1000:
            orientation = self.rng.integers(0, 2)
            nx = self.rng.uniform(0, self.total_length - l)
            ny = self.rng.uniform(0, self.total_width - w)
            if self._is_valid_placement(nx, ny, l, w, prev_fid, layout):
                layout[prev_fid] = (nx, ny, l, w, orientation)
                placed = True
            attempts += 1

        if not placed:
            raise RuntimeError(f"Failed to place connected machine {prev_fid} after multiple attempts.")

        # Place remaining connected machines
        for idx in range(1, len(self.connected_line)):
            fid = self.connected_line[idx]
            l, w = self.machine_dimensions[fid]

            placed = False
            for spacing in np.linspace(max_initial_spacing, max_expand_spacing, num=5):
                directions = ["right", "left", "up", "down"]
                self.rng.shuffle(directions)

                for direction in directions:
                    prev_x, prev_y, prev_l, prev_w, prev_o = layout[self.connected_line[idx - 1]]
                    prev_disp_l, prev_disp_w = (prev_w, prev_l) if prev_o == 1 else (prev_l, prev_w)

                    orientation = self.rng.integers(0, 2)
                    disp_l, disp_w = (w, l) if orientation == 1 else (l, w)

                    if direction == "right":
                        nx, ny = prev_x + prev_disp_l + spacing, prev_y + self.rng.uniform(-0.5, 0.5)
                    elif direction == "left":
                        nx, ny = prev_x - disp_l - spacing, prev_y + self.rng.uniform(-0.5, 0.5)
                    elif direction == "up":
                        nx, ny = prev_x + self.rng.uniform(-0.5, 0.5), prev_y + prev_disp_w + spacing
                    else:  # down
                        nx, ny = prev_x + self.rng.uniform(-0.5, 0.5), prev_y - disp_w - spacing

                    # Clip to bounds
                    nx = np.clip(nx, 0, self.total_length - disp_l)
                    ny = np.clip(ny, 0, self.total_width - disp_w)

                    # Validate
                    if self._is_valid_placement(nx, ny, disp_l, disp_w, fid, layout):
                        layout[fid] = (nx, ny, self.machine_dimensions[fid][0], self.machine_dimensions[fid][1], orientation)
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                # last resort: place anywhere valid
                for _ in range(50):
                    nx = self.rng.uniform(0, self.total_length - l)
                    ny = self.rng.uniform(0, self.total_width - w)
                    orientation = self.rng.integers(0, 2)
                    disp_l, disp_w = (w, l) if orientation == 1 else (l, w)
                    if self._is_valid_placement(nx, ny, disp_l, disp_w, fid, layout):
                        layout[fid] = (nx, ny, l, w, orientation)
                        placed = True
                        break
            if not placed:
                raise RuntimeError(f"Failed to place connected machine {fid} after multiple attempts.")

        # Now place remaining stations as usual
        for i in range(self.n_facilities):
            if i in self.connected_line:
                continue

            l, w = self.machine_dimensions[i]
            placed = False
            for _ in range(max_retries):
                x = self.rng.uniform(0, self.total_length - l)
                y = self.rng.uniform(0, self.total_width - w)
                orientation = self.rng.integers(0, 2)
                disp_l, disp_w = (w, l) if orientation == 1 else (l, w)

                if self._is_valid_placement(x, y, disp_l, disp_w, i, layout):
                    layout[i] = (x, y, l, w, orientation)
                    placed = True
                    break

            if not placed:
                raise RuntimeError(f"Failed to place facility {i} after {max_retries} attempts.")

        return layout

    # reset the environment at the beginning of placing facilities
    def reset(self, *, seed=None, options=None):
        # reset feature from gymnasium library taken
        super().reset(seed=seed)

        # self._select_random_flow_scenario()
        if self.fixed_scenario is not None:
            self.current_scenario = self.fixed_scenario
            self.flow_matrix = self.flow_scenarios[self.current_scenario]["matrix"]
        else:
            self._select_random_flow_scenario()

        # set everything to default after a reset
        self.current_facility_idx = 0
        self.current_layout = self._generate_random_layout()

        self.previous_cost = self._calculate_material_handling_cost()
        obs = self.get_observation()
        info = {"scenario": self.current_scenario}
        return obs, info

    # when the agent takes an action
    def step(self, action):
        self.step_count += 1

        # Unpack action
        dx, dy = np.clip(action[0:2], -1, 1)
        rotate = int(np.clip(np.round(action[2]), 0, 1))

        facility_id = self.current_facility_idx

        # Current machine state
        orig_length, orig_width = self.machine_dimensions[facility_id]
        x, y, stored_l, stored_w, orientation = self.current_layout[facility_id]

        # Apply rotation if chosen
        new_orientation = 1 - orientation if rotate == 1 else orientation
        if new_orientation == 1:
            l, w = orig_width, orig_length
        else:
            l, w = orig_length, orig_width

        # Proposed translation
        STEP_SIZE = 3.0
        new_x = np.clip(x + dx * STEP_SIZE, 0, self.total_length - l)
        new_y = np.clip(y + dy * STEP_SIZE, 0, self.total_width - w)

        # Compute cost before move
        cost_before = self._calculate_material_handling_cost()

        # Try move, revert if invalid
        if self._is_valid_placement(new_x, new_y, l, w, facility_id, self.current_layout):
            self.current_layout[facility_id] = (new_x, new_y, orig_length, orig_width, new_orientation)
            invalid_move = False
        else:
            invalid_move = True

        # Compute new cost
        cost_after = self._calculate_material_handling_cost()

        # Reward: scaled cost improvement
        reward = (cost_before - cost_after) / 1e6  # scale to manageable range

        # Small penalty for invalid actions
        if invalid_move:
            reward -= 0.1

        # Move to next facility (cyclic)
        self.current_facility_idx = (self.current_facility_idx + 1) % self.n_facilities

        # Check termination
        terminated = self.step_count >= self.max_steps
        truncated = False

        return self.get_observation(), reward, terminated, truncated, {"cost": cost_after}

    def _calculate_material_handling_cost(self):
        total_cost = 0.0

        # Get layout positions for all stations
        station_positions = {
            s: self.current_layout[i] for i, s in enumerate(self.station_references)
            if i in self.current_layout
        }

        for source in self.flow_matrix.index:
            for target in self.flow_matrix.columns:
                flow = self.flow_matrix.loc[source, target]
                if flow <= 0:
                    continue

                # Set the coordinates of the product source locations
                if str(source).startswith("E"):  # Entry point
                    if source not in self.entry_points_coordinates:
                        continue
                    sx, sy = self.entry_points_coordinates[source]

                elif str(source).startswith("S"):  # Station
                    if source not in station_positions:
                        continue
                    sx, sy = station_positions[source][0:2]

                else:
                    continue  # Unknown source type

                # Inter-station product flow
                if str(target).startswith("S"):
                    if target not in station_positions:
                        continue
                    tx, ty = station_positions[target][0:2]

                # S11 â†’ nearest exit
                elif str(source) in ["11", "s11"]:
                    # find nearest exit to S11
                    nearest_exit, min_dist = None, float("inf")
                    for exit_name, exit_data in self.exit_points.items():
                        ex, ey = exit_data["x"], exit_data["y"]
                        dist = abs(sx - ex) + abs(sy - ey)
                        if dist < min_dist:
                            nearest_exit, min_dist = exit_name, dist
                    tx, ty = self.exit_points[nearest_exit]["x"], self.exit_points[nearest_exit]["y"]

                else:
                    continue

                # Cost calculation based on distance
                dist = abs(sx - tx) + abs(sy - ty)
                total_cost += flow * dist

        return total_cost

    # check whether the distance between the parts of the packaging conveyor are max 2m placed apart
    def _check_conveyor_connectivity(self, layout=None):
        return True
        if layout is None:
            layout = self.current_layout
        max_distance = 2.0  # allowable distance between consecutive machines
        for i in range(len(self.connected_line) - 1):
            fid1 = self.connected_line[i]
            fid2 = self.connected_line[i + 1]

            x1, y1, l1, w1, o1 = layout[fid1]
            x2, y2, l2, w2, o2 = layout[fid2]

            # compute display dims per orientation
            dl1, dw1 = (w1, l1) if o1 == 1 else (l1, w1)
            dl2, dw2 = (w2, l2) if o2 == 1 else (l2, w2)

            cx1, cy1 = x1 + dl1 / 2.0, y1 + dw1 / 2.0
            cx2, cy2 = x2 + dl2 / 2.0, y2 + dw2 / 2.0

            distance = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
            if distance > max_distance:
                return False
        return True
    # checks the entire layout based on valid placements
    def _check_valid_layout(self):
        # Only check restricted zones and placement validity
        for fid, (x, y, orig_l, orig_w, orientation) in self.current_layout.items():
            if orientation == 1:
                l, w = orig_w, orig_l
            else:
                l, w = orig_l, orig_w

            if not self._is_valid_placement(x, y, l, w, fid, self.current_layout):
                return False

        # Skip conveyor connectivity check
        return True

    def get_observation(self):
        obs = []
        for fid in range(self.n_facilities):
            if fid in self.current_layout:
                x, y, _, _, _ = self.current_layout[fid]
            else:
                x, y = 0, 0
            obs.extend([x / self.total_length, y / self.total_width])

        # Add one-hot encoding of current facility being modified
        facility_indicator = np.zeros(self.n_facilities, dtype=np.float32)
        facility_indicator[self.current_facility_idx] = 1.0
        obs.extend(facility_indicator)

        return np.array(obs, dtype=np.float32)

    # creating the visualization
    def render(self):
        fig, ax = plt.subplots(figsize=(10, 8))

        # Draw irregular area outline
        irregular_outline = np.array([
            [0, 0],
            [29.1, 0],
            [29.1, 18.5],
            [37, 18.5],
            [37, 22.7],
            [29.1, 22.7],
            [29.1, 28.82],
            [29.1, 39.98],
            [12.04, 39.98],
            [12.04, 29.1],
            [0, 29.1],
            [0, 0]
        ])
        ax.plot(irregular_outline[:, 0], irregular_outline[:, 1], 'k-', lw=2)
        ax.fill(irregular_outline[:, 0], irregular_outline[:, 1], color='lightgray', alpha=0.3)

        # Draw restricted zones
        for (rx, ry, rl, rw) in self.restricted_zones:
            rect = Rectangle((rx, ry), rl, rw, color='red', alpha=0.5)
            ax.add_patch(rect)

        # Draw facilities with orientation
        colors = plt.cm.tab10(np.linspace(0, 1, self.n_facilities))
        for fid, (x, y, length, width, orientation) in self.current_layout.items():
            if orientation == 1:
                display_length, display_width = width, length
            else:
                display_length, display_width = length, width

            rect = Rectangle((x, y), display_length, display_width,
                             linewidth=1.5,
                             edgecolor='black',
                             facecolor=colors[fid % 10],
                             alpha=0.8)
            ax.add_patch(rect)

            cx, cy = x + display_length / 2.0, y + display_width / 2.0
            ax.text(cx, cy, str(fid), ha='center', va='center',
                    fontsize=10, color='black', fontweight='bold')
        # setting x, y axis of the figure
        ax.set_xlim(-1, self.total_length + 1)
        ax.set_ylim(-1, self.total_width + 1)
        ax.set_aspect('equal')
        ax.set_title(f"Irregular Layout - Step {self.step_count}")

        # When mode is set to rgb, the programme will internally draw rgb images that will be automatically closed again for computational reasons
        if self.render_mode == "rgb_array":
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            plt.close(fig)
            return img[:, :, :3]

        # When mode is set to human, the plots will be graphically presented to the user until it is manually closed
        elif self.render_mode == "human":
            plt.show()
            return None

# This class makes sure that the environment is made compatible to be used by the PPO agent
class PPOCompatibleEnv(gym.Env):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.observation_space = env.observation_space

        # flatten: [facility_id, dx, dy, rotate]
        self.action_space = spaces.Box(low=np.array([0, -1, -1, 0]), high=np.array([env.n_facilities - 1, 1, 1, 1]),
                                       dtype=np.float32)

    # Makes sure that all the keyword arguments from reset function are used again
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    # All the actions are clipped so that they are compatible with the PPO agent
    def step(self, action):
        # Clip and round actions to valid values
        dx, dy = np.clip(action[1:3], -1.0, 1.0)
        rotate = int(np.clip(np.round(action[3]), 0, 1))
        clipped_action = np.array([dx, dy, rotate], dtype=np.float32)
        return self.env.step(clipped_action)

    # Uses the previously defined render function
    def render(self):
        return self.env.render()

if __name__ == "__main__":

    print("[1] Loading Excel data...")
    sys.stdout.flush()

    # Call filepath again
    filepath = "C:\\Users\\beunk\\Downloads\\Bij-producten meta data.xlsx"

    try:
        shared_data = data_loader(filepath).excel_file_loading()
        print(f"[2] Excel data successfully loaded.")
        print(f"[3] Loaded Excel sheets: {list(shared_data.keys())}")
        sys.stdout.flush()
    except Exception as e:
        print(f"[ERROR] Failed to load Excel: {e}")
        sys.stdout.flush()
        raise

    # Loading environment
    print("[4] Loading predefined environment.")
    sys.stdout.flush()


    def make_env():
        stations_df = shared_data["Stations"]
        station_refs = [s for s in stations_df["Number"] if str(s).startswith("S")]

        base_env = ContinuousIrregularFLPEnv(
            n_facilities=len(station_refs),
            excel_data=shared_data,  # you called this 'shared_data', not 'preprocessed_data'
            render_mode=None,
            fixed_scenario=None,
            seed=None
        )
        return PPOCompatibleEnv(base_env)


    print("[5] Predefined environment loaded.")
    sys.stdout.flush()


    # Create dummy environments
    print("[6] Creating 6 parallel environments.")
    sys.stdout.flush()

    try:
        # Preferred parallelization
        vec_env = SubprocVecEnv([make_env for _ in range(6)])
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_reward=10.0)
        print("[7] SubprocVecEnv successfully created")
    except Exception as e:
        print("[7] SubprocVecEnv failed on Windows, switching to DummyVecEnv")
        sys.stdout.flush()

        # Fallback: DummyVecEnv always works
        vec_env = DummyVecEnv([make_env for _ in range(6)])
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_reward=10.0)


    # Call the PPO model with affiliated hyperparameters
    print("[8] Creating PPO model with TensorBoard logging...")
    sys.stdout.flush()

    # Create timestamped directories for TensorBoard logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tensorboard_dir = f"./tensorboard_logs/ppo_training_{timestamp}"

    os.makedirs(tensorboard_dir, exist_ok=True)

    try:
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,  # Changed from 0 to 1 for more output
            device="cuda",
            batch_size=512,
            n_steps=2048,
            learning_rate=3e-4,
            n_epochs=5,
            gamma=0.99,
            clip_range=0.2,
            ent_coef=0.2,
            tensorboard_log=tensorboard_dir,
        )
        print("[9] PPO model successfully created")
        print(f"[9.1] TensorBoard logs: tensorboard --logdir={tensorboard_dir}")
        sys.stdout.flush()
    except Exception as e:
        print(f"[ERROR] Failed to create PPO model: {e}")
        sys.stdout.flush()
        raise

    # Training of the model
    print("[10] Starting training")
    print("[11] Training results will be logged to: ppo_training_log##.csv")
    sys.stdout.flush()

    try:
        csv_callback = EnhancedLogging(csv_filename="ppo_training_log70.csv", tensorboard_log=tensorboard_dir)

        model.learn(
            total_timesteps=1000000, # Important for how long you want to train the model for
            callback=csv_callback,
            log_interval=10
        )
        print("[12] Model was trained succesful")
        sys.stdout.flush()
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        sys.stdout.flush()
        raise

    # Saving of the model on the internal drive
    print("[13] Saving trained model on internal drive")
    sys.stdout.flush()

    try:
        model.save("ppo_flp_agent")
        vec_env.save("ppo_flp_agent_vecnorm.pkl")
        print("âœ“ Saved VecNormalize statistics")
        print("[14] Model saved as 'ppo_flp_agent'")
        sys.stdout.flush()
    except Exception as e:
        print(f"[ERROR] Failed to save model: {e}")
        sys.stdout.flush()
        raise

    print("\n" + "=" * 60)
    print("Training is completed")
    print("=" * 60)
    print("\nðŸ“Š Results saved to:")
    print("  âœ“ ppo_training_log##.csv")
    print("=" * 60)

    # testing the trained model
    print("\n" + "=" * 60)
    print("[15] Initiating the testing phase of the trained model.")
    print("=" * 60)

    # Load the trained model
    print("[16] Loading trained model and normalization statistics")
    try:
        model = PPO.load("C:/Users/beunk/PycharmProjects/PythonProject1/ppo_flp_agent.zip")
        print("[17] Model loaded successfully.")
        sys.stdout.flush()
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.stdout.flush()
        raise

    def compute_machine_distances(layout):
        distances = []
        ids = sorted(layout.keys())

        # Precompute centers
        centers = {}
        for fid, (x, y, l, w, o) in layout.items():
            dl, dw = (w, l) if o == 1 else (l, w)
            cx, cy = x + dl / 2.0, y + dw / 2.0
            centers[fid] = (cx, cy)

        # Compute pairwise Manhattan distances
        for i, j in itertools.combinations(ids, 2):
            (x1, y1), (x2, y2) = centers[i], centers[j]
            dist = abs(x1 - x2) + abs(y1 - y2)
            distances.append((i, j, dist))

        return distances


    def test_with_warm_starts(num_episodes=2000, max_steps=200):
        print(f"\n[18] Testing with warm-start optimization ({num_episodes} episodes)...")
        sys.stdout.flush()

        stations_df = shared_data["Stations"]
        station_refs = [s for s in stations_df["Number"] if str(s).startswith("S")]

        def make_test_env():
            base_env = ContinuousIrregularFLPEnv(
                n_facilities=len(station_refs),
                excel_data=shared_data,
                render_mode="rgb_array",
                fixed_scenario="regular",
                max_steps=max_steps
            )
            return PPOCompatibleEnv(base_env)

        test_vec_env = DummyVecEnv([make_test_env])

        try:
            test_vec_env = VecNormalize.load("ppo_flp_agent_vecnorm.pkl", test_vec_env)
            test_vec_env.training = False
            test_vec_env.norm_reward = False
            print("[18.1] VecNormalize statistics loaded")
        except Exception as e:
            print(f"[WARNING] Could not load VecNormalize: {e}")

        results = []
        best_cost_seen = float('inf')
        best_layout_seen = None

        for episode in range(num_episodes):
            obs = test_vec_env.reset()
            episode_costs = []

            # Run full episode with deterministic policy
            for step in range(max_steps):
                action, _ = model.predict(obs, deterministic=False)
                obs, reward, done, info = test_vec_env.step(action)

                current_cost = info[0].get('cost', 0)
                episode_costs.append(current_cost)

                if done[0]:
                    break

            # Record the BEST cost achieved during this episode
            best_episode_cost = min(episode_costs) if episode_costs else episode_costs[-1]
            final_layout = test_vec_env.envs[0].env.current_layout.copy()

            if best_episode_cost < best_cost_seen:
                best_cost_seen = best_episode_cost
                best_layout_seen = final_layout.copy()

            results.append({
                'episode': episode + 1,
                'layout': final_layout,
                'final_cost': best_episode_cost,
                'total_reward': sum(episode_costs)
            })

            if (episode + 1) % 10 == 0:
                print(f"[19] Episode {episode + 1}/{num_episodes}. Best so far: {best_cost_seen / 1_000_000:.2f}M")
                sys.stdout.flush()

        test_vec_env.close()
        results.sort(key=lambda x: x['final_cost'])
        return results


    # test the model based on the scenario: num_episodes important as to how many times the model is tested
    def test_on_scenario(num_episodes=400000, max_steps=200):
        print(f"\n[18] Testing on regular flow scenario ({num_episodes} episodes)...")
        print("[18.1] Tracking ONLY FINAL layout costs per episode")
        sys.stdout.flush()

        stations_df = shared_data["Stations"]
        station_refs = [s for s in stations_df["Number"] if str(s).startswith("S")]

        # Create base environment
        def make_test_env():
            base_env = ContinuousIrregularFLPEnv(
                n_facilities=len(station_refs),
                excel_data=shared_data,
                render_mode="rgb_array",
                fixed_scenario="regular",
                max_steps=max_steps
            )
            return PPOCompatibleEnv(base_env)

        # Wrap in DummyVecEnv (single environment for testing)
        test_vec_env = DummyVecEnv([make_test_env])

        # Load the saved VecNormalize statistics from training
        try:
            test_vec_env = VecNormalize.load("ppo_flp_agent_vecnorm.pkl", test_vec_env)
            # CRITICAL: Set training=False and norm_reward=False for testing
            test_vec_env.training = False
            test_vec_env.norm_reward = False
            print("[18.2] VecNormalize statistics loaded successfully")
        except Exception as e:
            print(f"[WARNING] Could not load VecNormalize stats: {e}")
            print("[18.3] Continuing without normalization (results may be suboptimal)")

        results = []

        for episode in range(num_episodes):
            obs = test_vec_env.reset()
            episode_reward = 0
            done = False
            step_count = 0

            # Run through entire episode WITHOUT tracking intermediate costs
            while not done and step_count < max_steps:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = test_vec_env.step(action)

                episode_reward += reward[0]
                step_count += 1

                # Check if episode terminated
                done = done[0]

            # ONLY calculate cost ONCE at the end of the episode
            final_cost = test_vec_env.envs[0].env._calculate_material_handling_cost()
            final_layout = test_vec_env.envs[0].env.current_layout.copy()

            results.append({
                'episode': episode + 1,
                'layout': final_layout,
                'final_cost': final_cost,  # This is the ONLY cost we track
                'total_reward': episode_reward
            })

            if (episode + 1) % 50 == 0:  # Changed from 20 to 50 for less output
                avg_cost = np.mean([r['final_cost'] for r in results[-50:]])
                print(f"[19] Episode {episode + 1}/{num_episodes} | "
                      f"Last 50 avg: {avg_cost / 1_000_000:.2f}M")
                sys.stdout.flush()

        test_vec_env.close()

        # Sort results by final cost (ascending)
        results.sort(key=lambda x: x['final_cost'])

        print("\n" + "=" * 60)
        print(f"[20] Testing complete - tracked {num_episodes} FINAL layout costs")
        print("=" * 60)
        return results


    # Test on all scenarios
    regular_flow_results = test_on_scenario(num_episodes=400000, max_steps=200)


    # Statistical summary
    costs = [r['final_cost'] for r in regular_flow_results]
    rewards = [r['total_reward'] for r in regular_flow_results]

    print("\n [20]Statistical Summary:")
    print(f"Costs (in millions): Mean = {np.mean(costs)/1_000_000:.2f}, Median = {np.median(costs)/1_000_000:.2f}, "
          f"Std = {np.std(costs)/1_000_000:.2f}, Min = {np.min(costs)/1_000_000:.2f}, Max = {np.max(costs)/1_000_000:.2f}")
    print(f"Rewards: Mean = {np.mean(rewards):.4f}, Median = {np.median(rewards):.4f}, "
          f"Std = {np.std(rewards):.4f}, Min = {np.min(rewards):.4f}, Max = {np.max(rewards):.4f}")


    # Visualizing top 5 results graphically
    print("\n" + "=" * 60)
    print("Visualizing top 5 results graphically")
    print("=" * 60)


    def render_layout_human(scenario_name, layout, final_cost, total_reward, env, rank=None):
        print(f"\n[21] Displaying layout for regular flow scenario (Rank #{rank})...")
        print(f"[21] Final Cost: {final_cost / 1_000_000:.4f}M | Total Reward: {total_reward:.4f}")
        sys.stdout.flush()

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))

        # Draw irregular area outline
        irregular_outline = np.array([
            [0, 0],
            [29.1, 0],
            [29.1, 18.5],
            [37, 18.5],
            [37, 22.7],
            [29.1, 22.7],
            [29.1, 28.82],
            [29.1, 39.98],
            [12.04, 39.98],
            [12.04, 29.1],
            [0, 29.1],
            [0, 0]
        ])
        ax.plot(irregular_outline[:, 0], irregular_outline[:, 1], 'k-', lw=2)
        ax.fill(irregular_outline[:, 0], irregular_outline[:, 1], color='lightgray', alpha=0.3)

        # Draw restricted zones
        for (rx, ry, rw, rh) in env.restricted_zones:
            rect = Rectangle((rx, ry), rw, rh, color='red', alpha=0.5)
            ax.add_patch(rect)

        # Draw facilities with orientation
        colors = plt.cm.tab10(np.linspace(0, 1, env.n_facilities))  # type: ignore
        for fid, (x, y, stored_w, stored_h, orientation) in layout.items():
            if orientation == 1:
                display_w, display_h = stored_h, stored_w
            else:
                display_w, display_h = stored_w, stored_h

            rect = Rectangle((x, y), display_w, display_h,
                             linewidth=2,
                             edgecolor='black',
                             facecolor=colors[fid % 10],
                             alpha=0.8)
            ax.add_patch(rect)

            cx, cy = x + display_w / 2, y + display_h / 2
            ax.text(cx, cy, str(fid), ha='center', va='center',
                    fontsize=12, color='black', fontweight='bold')

        ax.set_xlim(-2, env.total_length + 2)
        ax.set_ylim(-2, env.total_width + 2)
        ax.set_aspect('equal')
        ax.set_xlabel('Length (m)', fontsize=12)
        ax.set_ylabel('Width (m)', fontsize=12)
        ax.set_title(
            f"Top #{rank} - Facility Layout ({scenario_name.upper()} Flow)\n"
            f"Cost: {final_cost / 1_000_000:.2f}M | Reward: {total_reward:.4f}",
            fontsize=14, fontweight='bold'
        )
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


    # Render regular flow results

    print("\n[22] REGULAR FLOW SCENARIO RESULTS:")
    stations_df = shared_data["Stations"]
    station_refs = [s for s in stations_df["Number"] if str(s).startswith("S")]
    regular_env = ContinuousIrregularFLPEnv(
        n_facilities=len(station_refs),
        excel_data=shared_data,
        render_mode="human",
        fixed_scenario="regular"
    )

    # Sort and select top 5 by lowest cost
    top5 = sorted(regular_flow_results, key=lambda x: x['final_cost'])[:5]

    # Print + visualize
    for rank, result in enumerate(top5, start=1):
        print("\n" + "-" * 60)
        print(f"TOP #{rank} LAYOUT")
        print("-" * 60)
        print(f"Final Cost: {result['final_cost'] / 1_000_000:.4f}M")
        print(f"Total Reward: {result['total_reward']:.4f}")
        print("Machine Coordinates (x, y, w, h, orientation):")

        for fid, (x, y, w, h, orientation) in result['layout'].items():
            print(f"  S{fid}: x={x:.2f}, y={y:.2f}, w={w:.2f}, h={h:.2f}, orient={orientation}")

        # Visualize
        render_layout_human(
            "regular",
            result['layout'],
            result['final_cost'],
            result['total_reward'],
            regular_env,
            rank=rank
        )

    print("\n" + "=" * 60)
    print("[23] Pairwise Manhattan distances for best layout")
    print("=" * 60)

    # Get the best layout (lowest total cost) from your test results
    regular_flow_results.sort(key=lambda x: x['final_cost'])
    best_layout = regular_flow_results[0]['layout']
    best_cost = regular_flow_results[0]['final_cost']

    # Use the Manhattan distance function
    pairwise_distances = compute_machine_distances(best_layout)

    # Print all distances
    for i, j, dist in pairwise_distances:
        print(f"Distance between Machine {i} and Machine {j}: {dist:.2f}")

    print("=" * 60)
    total_distance = sum(dist for _, _, dist in pairwise_distances)
    print(f"Total machine distances: {total_distance:.2f}")
    print(f"Best layout total cost: {best_cost:.2f}")
    print("=" * 60)

    # Summary
    print("\nSummary:")
    print(f"Training completed on stochastic flow scenario")
    print(f"Tested on 'regular' flow scenario: {len(regular_flow_results)} episodes")
