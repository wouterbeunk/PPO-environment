# TRAINING SCRIPT WITH CSV LOGGING

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
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import sys
import csv
from datetime import datetime
from stable_baselines3.common.vec_env import VecNormalize
import random

# CSV LOGGING CALLBACK - Exports metrics to CSV file
class CSVLoggingCallback(BaseCallback):
    """
    Callback that logs metrics to a CSV file during training.
    """

    def __init__(self, csv_filename="ppo_training_log.csv", verbose=0):
        super().__init__(verbose)
        self.csv_filename = csv_filename
        self.header_written = False
        self.step_count = 0

        # Create CSV file with header
        print(f"[CSV] Logging to: {csv_filename}")
        with open(self.csv_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestep", "Cost", "Reward", "Episode", "Timestamp"])
        print(f"[CSV] ✓ CSV file initialized")

    def _on_step(self) -> bool:
        self.step_count += 1
        infos = self.locals.get('infos', [])
        rewards = self.locals.get('rewards', None)

        # Defensive: if rewards is a scalar or None, wrap into list
        if rewards is None:
            rewards_list = []
        else:
            try:
                rewards_list = list(rewards)
            except Exception:
                rewards_list = [float(rewards)]

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

            if cost is not None:
                cost = cost / 10000.0

            if cost is not None or reward is not None:
                with open(self.csv_filename, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        self.step_count,
                        f"{cost:.4f}" if cost is not None else "",
                        f"{reward:.4f}" if reward is not None else "",
                        env_idx,
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    ])
        return True


# ============================================================
# STEP 0: DEBUG SETUP
# ============================================================
print("[DEBUG] Script starting...")
sys.stdout.flush()


# ============================================================
# YOUR ENVIRONMENT CLASS
# ============================================================
class ContinuousIrregularFLPEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self,
                 n_facilities,
                 max_steps: int = 200,
                 render_mode: str = "human",
                 seed: int = 42,
                 excel_data=None,
                 fixed_scenario=None):

        super().__init__()

        np.random.seed(seed)

        self.max_steps = max_steps
        self.render_mode = render_mode

        self.main_area = (0, 0, 29.0, 28.82)
        self.top_right_area = (12.04, 29.1, 17.06, 10.88)
        self.mid_right_area = (29.1, 18.5, 7.9, 4.2)

        self.total_length = 37.0
        self.total_width = 39.98

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

        # --- Load Excel data ---
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

        # Now use `extracted_data` for all your DataFrame setup
        stations_df = extracted_data["Stations"]
        high_flow_df = extracted_data["High flow matrix"]
        regular_flow_df = extracted_data["Regular flow matrix"]
        low_flow_df = extracted_data["Low flow matrix"]
        entry_points_df = extracted_data["Entry Points"]

        # creating variables to link the machines of both sheets with each other
        station_refs = [s for s in stations_df["Number"] if str(s).startswith("S")]
        entry_refs = [idx for idx in regular_flow_df.index if str(idx).startswith("E")]

        # fills all the non values with 0s in the flow matrices
        def clean_flow(df):
            return df.loc[
                [r for r in df.index if str(r).startswith(("S", "E"))],
                [c for c in df.columns if str(c).startswith("S")]
            ].fillna(0).astype(float)

        # set the flow clean flow matrices
        flow_high_df = clean_flow(high_flow_df)
        flow_regular_df = clean_flow(regular_flow_df)
        flow_low_df = clean_flow(low_flow_df)

        # Store the flow scenarios as a dictionary
        self.flow_scenarios = {
            "high": {"matrix": flow_high_df, "weight": 1 / 12},
            "regular": {"matrix": flow_regular_df, "weight": 9 / 12},
            "low": {"matrix": flow_low_df, "weight": 2 / 12}, }

        # set default flow matrix to regular
        self.flow_matrix = flow_regular_df
        self.fixed_scenario = fixed_scenario

        # Store the station and entry references of the excel sheets: E.g. E0 and S0
        self.station_references = station_refs
        self.entry_references = entry_refs

        # set dimensions
        self.machine_dimensions = stations_df[["Length", "Width"]].values.tolist()
        self.n_facilities = len(station_refs)

        # setting the product flow entry point coordinates from the excel file
        self.entry_points_coordinates = {}
        for _, row in entry_points_df.iterrows():
            entry_point = row["EntryPoint"]
            x = float(row["X"])
            y = float(row["Y"])

            # format the entry point dictionary to Station: {entry point, flow}
            self.entry_points_coordinates[entry_point] = (x, y)

        # S8-S11 are seperate parts of the packaging machines. Creating an additional parameter for close placement
        self.connected_line = [8, 9, 10, 11]
        self.connected_line_spacing = 0.0

        # system exit points: when by-products leave the department to storage/customer (two doors they could leave through)
        self.exit_points = {
            "Exit1": {"x": 0.0, "y": 12.5},
            "Exit2": {"x": 0.0, "y": 26.4},
        }
        # creating the action space dictionary, because facilityID is discrete and placement should be continuous
        # both x and y coordinates of the facilities can be moved in a continuous action space ranging from -1 until 1
        # agent can choose whether to not rotate = 0 or rotate = 1
        self.action_space = spaces.Dict({
            'facility_id': spaces.Discrete(n_facilities),
            'move': spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            'rotate': spaces.Discrete(2)
        })

        # setting the total possible area where the agent can look (so from 0, 0 until the max length and width)
        self.observation_space = spaces.Box(
            low=np.float32(0.0),
            high=np.float32(max(self.total_length, self.total_width)),
            shape=(self.n_facilities * 2,),
            dtype=np.float32
        )

        # Initializing variables
        self.current_layout = None
        self.step_count = 0
        self.previous_cost = float('inf')
        # random number generation based on the seed
        self.rng = np.random.default_rng(seed)

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

        corners = [(x, y), (x + l, y), (x, y + w), (x + l, y + w)]
        for (px, py) in corners:
            if not (inside_main(px, py) or inside_top_right(px, py) or inside_mid_right(px, py)):
                return False
        return True

    # creating a function for restricted zones
    # x+l is cannot overlap with the x of restricted zone for example
    def _is_in_restricted_zone(self, x, y, l, w):
        for (rx, ry, rl, rw) in self.restricted_zones:
            if not (x + l <= rx or rx + rl <= x or y + w <= ry or ry + rw <= y):
                return True
        return False

    # combining previous function and determining whether the placement is valid or not
    def _is_valid_placement(self, x, y, length, width, facility_id, layout=None):
        """
        Check placement inside allowed areas, not in restricted zones, and no overlap.
        layout entries are canonical: (x, y, length, width, orientation)
        """
        if not self._is_in_irregular_area(x, y, length, width):
            return False
        if self._is_in_restricted_zone(x, y, length, width):
            return False

        if layout is None:
            layout = self.current_layout

        for other_id, (ox, oy, ol, ow, o_orientation) in layout.items():
            if other_id == facility_id:
                continue

            # Determine display dimensions of other facility
            if o_orientation == 1:
                other_display_length, other_display_width = ow, ol
            else:
                other_display_length, other_display_width = ol, ow

            # Axis-aligned rectangle overlap test
            if not (x + length <= ox or ox + other_display_length <= x or
                    y + width <= oy or oy + other_display_width <= y):
                return False

        return True

    # this function calculates the distances of all machines on average to restricted zones
    # needs to be as low as possible
    def _calculate_proximity_to_restricted_areas(self):
        rz = np.array(self.restricted_zones)  # shape (N_zones, 4)
        total_distance = 0.0
        for fid, (x, y, length, width, orientation) in self.current_layout.items():
            # compute display dims
            dl, dw = (width, length) if orientation == 1 else (length, width)
            cx, cy = x + dl / 2.0, y + dw / 2.0

            # distances to each restricted zone
            closest_xs = np.clip(cx, rz[:, 0], rz[:, 0] + rz[:, 2])
            closest_ys = np.clip(cy, rz[:, 1], rz[:, 1] + rz[:, 3])
            dists = np.sqrt((cx - closest_xs) ** 2 + (cy - closest_ys) ** 2)
            min_dist = np.min(dists)
            total_distance += float(min_dist)

        if len(self.current_layout) == 0:
            return float('inf')
        avg_distance = total_distance / len(self.current_layout)
        return avg_distance

    # function creates everytime a random layout for the learning agent
    def _generate_random_layout(self):
        layout = {}
        max_retries = 1000

        # The following lines make sure that S8-S11 are placed relatively close to each other
        bbox_margin = 3.0  # maximum distance allowed between consecutive machines
        # total_l and total_w sums the total length and width of the selected machines
        total_l = sum(self.machine_dimensions[fid][0] for fid in self.connected_line)
        total_w = sum(self.machine_dimensions[fid][1] for fid in self.connected_line)
        # makes sure the entire line is not longer than the total area length and width
        line_bbox_l = min(self.total_length, total_l + bbox_margin * (len(self.connected_line) - 1))
        line_bbox_w = min(self.total_width, total_w + bbox_margin * (len(self.connected_line) - 1))

        # Random origin of the connected line bounding box
        box_x = self.rng.uniform(0, self.total_length - line_bbox_l)
        box_y = self.rng.uniform(0, self.total_width - line_bbox_w)

        for idx, fid in enumerate(self.connected_line):
            orig_l, orig_w = self.machine_dimensions[fid]
            placed = False

            for _ in range(max_retries):
                # Place each machine randomly inside the bounding box
                x = self.rng.uniform(box_x, min(box_x + line_bbox_l - orig_l, self.total_length - orig_l))
                y = self.rng.uniform(box_y, min(box_y + line_bbox_w - orig_w, self.total_width - orig_w))

                # check the orientation
                for orientation in [0, 1]:
                    w, h = (orig_w, orig_l) if orientation == 1 else (orig_l, orig_w)
                    temp_layout = layout.copy()
                    temp_layout[fid] = (x, y, orig_l, orig_w, orientation)
                    # check whether it is a valid placement
                    if self._is_valid_placement(x, y, w, h, fid, temp_layout):
                        layout[fid] = (x, y, orig_l, orig_w, orientation)
                        placed = True
                        break

                if placed:
                    break

            if not placed:
                raise RuntimeError(f"Failed to place connected facility {fid} after {max_retries} attempts.")

        # --- Place remaining machines ---
        for i in range(self.n_facilities):
            if i in self.connected_line:
                continue  # already placed

            orig_l, orig_w = self.machine_dimensions[i]
            placed = False

            for _ in range(max_retries):
                x = self.rng.uniform(0, self.total_length - orig_l)
                y = self.rng.uniform(0, self.total_width - orig_w)

                for orientation in [0, 1]:
                    l, w = (orig_w, orig_l) if orientation == 1 else (orig_l, orig_w)
                    temp_layout = layout.copy()
                    temp_layout[i] = (x, y, orig_l, orig_w, orientation)

                    if self._is_valid_placement(x, y, l, w, i, temp_layout):
                        layout[i] = (x, y, orig_l, orig_w, orientation)
                        placed = True
                        break

                if placed:
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
        else:
            self.current_scenario = random.choice(list(self.flow_scenarios.keys()))

        # set everything to default after a reset
        self.step_count = 0
        self.current_layout = self._generate_random_layout()
        self.previous_cost = self._calculate_material_handling_cost()
        obs = self._get_observation()
        info = {"scenario": self.current_scenario}
        return obs, info

    # when the agent takes an action
    def step(self, action):
        self.step_count += 1
        facility_id = action['facility_id']
        dx, dy = np.clip(action['move'], -1.0, 1.0)
        rotate = action.get('rotate', 0)

        # Get original dimensions from machine_dimensions (these never change)
        orig_length, orig_width = self.machine_dimensions[facility_id]
        x, y, stored_l, stored_w, orientation = self.current_layout[facility_id]

        # Apply rotation action
        if rotate == 1:
            orientation = 1 - orientation

        # Get CURRENT dimensions based on orientation (computed, not stored)
        if orientation == 1:
            l, w = orig_width, orig_length  # Swapped
        else:
            l, w = orig_length, orig_width  # Normal

        new_x = np.clip(x + dx, 0, self.total_length - l)
        new_y = np.clip(y + dy, 0, self.total_width - w)

        # Create temporary layout - ALWAYS store original dimensions!
        temp_layout = self.current_layout.copy()
        temp_layout[facility_id] = (new_x, new_y, orig_length, orig_width, orientation)

        # Validate placement using the current dimensions (w, h)
        if self._is_valid_placement(new_x, new_y, l, w, facility_id, temp_layout):
            self.current_layout[facility_id] = (new_x, new_y, orig_length, orig_width, orientation)

        # If placement is invalid, keep the old position but still apply rotation if it was just a rotation
        elif rotate == 1:
            # Allow rotation in place even if movement was invalid
            self.current_layout[facility_id] = (x, y, orig_length, orig_width, orientation)

        # check whether the box and bucket line are closely placed
        if facility_id in self.connected_line:
            idx = self.connected_line.index(facility_id)
            # Move all following machines relative to this one
            for i in range(idx + 1, len(self.connected_line)):
                next_fid = self.connected_line[i]
                prev_fid = self.connected_line[i - 1]
                px, py, pw, ph, po = self.current_layout[prev_fid]
                pw_disp, ph_disp = (ph, pw) if po == 1 else (pw, ph)
                # Place next machine adjacent horizontally
                self.current_layout[next_fid] = (px + pw_disp + self.connected_line_spacing,
                                                 py,
                                                 self.machine_dimensions[next_fid][0],
                                                 self.machine_dimensions[next_fid][1],
                                                 self.current_layout[next_fid][4])

        # Reward calculation
        cost = self._calculate_material_handling_cost()
        cost_reward = (self.previous_cost - cost) / (abs(self.previous_cost) + 1e-6)
        cost_reward = np.clip(cost_reward * 5.0, -1.0, 1.0)  # scale improvements for PPO

        avg_dist_to_restricted = self._calculate_proximity_to_restricted_areas()
        proximity_reward = 1 / (1 + avg_dist_to_restricted)
        proximity_reward = (proximity_reward - 0.5) * 2  # normalize roughly to [−1, 1]
        proximity_weight = 0.01
        total_reward = cost_reward + proximity_weight * (proximity_reward - 0.5)

        if not self._check_valid_layout():
            total_reward -= 0.5

        self.previous_cost = cost

        terminated = self.step_count >= self.max_steps
        truncated = False
        return self._get_observation(), total_reward, terminated, truncated, {'cost': cost}

    def _calculate_material_handling_cost(self):
        total_cost = 0.0

        # Get all layout positions for stations
        station_positions = {
            s: self.current_layout[i] for i, s in enumerate(self.station_references)
            if i in self.current_layout
        }

        # Iterate over all flows (E→S and S↔S)
        for source in self.flow_matrix.index:
            for target in self.flow_matrix.columns:
                flow = self.flow_matrix.loc[source, target]
                if flow <= 0:
                    continue

                # Source coordinates
                if str(source).startswith("E"):  # Entry point source
                    if source not in self.entry_points_coordinates:
                        continue  # skip missing entry
                    sx, sy = self.entry_points_coordinates[source]
                elif str(source).startswith("S"):  # Station source
                    if source not in station_positions:
                        continue
                    sx, sy = station_positions[source][0:2]
                else:
                    continue  # skip unknown labels

                # Target coordinates
                if target not in station_positions:
                    continue
                tx, ty = station_positions[target][0:2]

                # Compute Manhattan distance
                dist = abs(sx - tx) + abs(sy - ty)

                # Add cost
                total_cost += flow * dist

        return total_cost

    # check whether the distance between the parts of the packaging conveyor are max 3m placed apart
    def _check_conveyor_connectivity(self, layout=None):
        if layout is None:
            layout = self.current_layout
        max_distance = 3.0  # allowable distance between consecutive machines
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
        # Validate each placement by passing canonical length,width
        for fid, (x, y, length, width, orientation) in self.current_layout.items():
            if not self._is_valid_placement(x, y, length, width, fid, self.current_layout):
                return False
        # Check conveyor connectivity once
        if not self._check_conveyor_connectivity(self.current_layout):
            return False
        return True

    def _get_observation(self):
        obs = []
        for fid in range(self.n_facilities):
            if fid in self.current_layout:
                x, y, _, _, _ = self.current_layout[fid]
            else:
                x, y = 0, 0
            obs.extend([x / self.total_length, y / self.total_width])
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
        for (rx, ry, rw, rh) in self.restricted_zones:
            rect = Rectangle((rx, ry), rw, rh, color='red', alpha=0.5)
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

        ax.set_xlim(-1, self.total_length + 1)
        ax.set_ylim(-1, self.total_width + 1)
        ax.set_aspect('equal')
        ax.set_title(f"Irregular Layout - Step {self.step_count}")

        if self.render_mode == "rgb_array":
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            plt.close(fig)
            return img[:, :, :3]

        elif self.render_mode == "human":
            plt.show()
            return None


class PPOCompatibleEnv(gym.Env):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.observation_space = env.observation_space
        # flatten: [facility_id, dx, dy, rotate]
        self.action_space = spaces.Box(low=np.array([0, -1, -1, 0]), high=np.array([env.n_facilities - 1, 1, 1, 1]),
                                       dtype=np.float32)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        # Clip and round actions to valid values
        facility_id = int(np.clip(np.round(action[0]), 0, self.env.n_facilities - 1))
        dx, dy = np.clip(action[1:3], -1.0, 1.0)
        rotate = int(np.clip(np.round(action[3]), 0, 1))
        dict_action = {'facility_id': facility_id, 'move': np.array([dx, dy]), 'rotate': rotate}
        return self.env.step(dict_action)

    def render(self):
        return self.env.render()


print("[STEP 1] Loading Excel data...")
sys.stdout.flush()

filepath = "C:\\Users\\beunk\\Downloads\\Bij-producten meta data.xlsx"

try:
    shared_data = data_loader(filepath).excel_file_loading()
    print(f"[STEP 1] ✓ Excel data loaded successfully!")
    print(f"[STEP 1] Keys in data: {list(shared_data.keys())}")
    sys.stdout.flush()
except Exception as e:
    print(f"[ERROR] Failed to load Excel: {e}")
    sys.stdout.flush()
    raise

# ============================================================
# STEP 2: ENVIRONMENT FACTORY (No duplicate loading!)
# ============================================================
print("[STEP 2] Creating environment factory...")
sys.stdout.flush()


def make_env():
    stations_df = shared_data["Stations"]
    station_refs = [s for s in stations_df["Number"] if str(s).startswith("S")]

    base_env = ContinuousIrregularFLPEnv(
        n_facilities=len(station_refs),
        excel_data=shared_data,  # you called this 'shared_data', not 'preprocessed_data'
        render_mode="rgb_array",
        fixed_scenario="regular"  # fixed for training only
    )
    return PPOCompatibleEnv(base_env)


print("[STEP 2] ✓ Environment factory created")
sys.stdout.flush()

# ============================================================
# STEP 3: CREATE VECTORIZED ENVIRONMENT
# ============================================================
print("[STEP 3] Creating 4 vectorized environments...")
sys.stdout.flush()

try:
    print("[STEP 3] Creating 3 DummyVecEnv environments...")
    vec_env = DummyVecEnv([make_env for _ in range(2)])
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=1.0) # type: ignore
    print("[STEP 3] ✓ DummyVecEnv created!")
    sys.stdout.flush()
except Exception as e:
    print(f"[ERROR] Failed to create vectorized environment: {e}")
    sys.stdout.flush()
    raise


# STEP 4: CREATE PPO AGENT
print("[STEP 4] Creating PPO model...")
sys.stdout.flush()

try:
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        batch_size=512,
        n_steps=4096,
        learning_rate=3e-4,
        n_epochs=3,
        gamma=0.99,
        clip_range=0.3,
        ent_coef=0.005
    )
    print("[STEP 4] ✓ PPO model created successfully!")
    sys.stdout.flush()
except Exception as e:
    print(f"[ERROR] Failed to create PPO model: {e}")
    sys.stdout.flush()
    raise

# STEP 5: TRAIN THE MODEL WITH CSV LOGGING
print("[STEP 5] Starting training")
print("[STEP 5] Logging to: ppo_training_log.csv")
sys.stdout.flush()

try:
    # Create both callbacks: TensorBoard + CSV
    csv_callback = CSVLoggingCallback(csv_filename="ppo_training_log23.csv")

    model.learn(
        total_timesteps=250000,
        callback=csv_callback,
        log_interval=10
    )
    print("[STEP 5] ✓ Training completed!")
    sys.stdout.flush()
except Exception as e:
    print(f"[ERROR] Training failed: {e}")
    sys.stdout.flush()
    raise

# STEP 6: SAVE THE MODEL
print("[STEP 6] Saving trained model...")
sys.stdout.flush()

try:
    model.save("ppo_flp_agent")
    print("[STEP 6] ✓ Model saved as 'ppo_flp_agent'")
    sys.stdout.flush()
except Exception as e:
    print(f"[ERROR] Failed to save model: {e}")
    sys.stdout.flush()
    raise

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print("\n📊 Results saved to:")
print("  ✓ ppo_training_log.csv (metrics at each step)")
print("=" * 60)

# PHASE 2: TESTING ON DIFFERENT FLOW SCENARIOS
print("\n" + "=" * 60)
print("PHASE 2: TESTING ON HIGH AND LOW FLOW SCENARIOS")
print("=" * 60)

# Load the trained model
print("[TEST] Loading trained model...")
try:
    model = PPO.load("ppo_flp_agent")
    print("[TEST] ✓ Model loaded successfully!")
    sys.stdout.flush()
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    sys.stdout.flush()
    raise


def test_on_scenario(scenario_name, num_episodes=160, max_steps=200):
    """
    Test the trained model on a specific flow scenario.
    Returns a sorted list of results by final_cost (ascending).
    """
    print(f"\n[TEST] Testing on '{scenario_name}' flow scenario ({num_episodes} episodes)...")
    sys.stdout.flush()

    stations_df = shared_data["Stations"]
    station_refs = [s for s in stations_df["Number"] if str(s).startswith("S")]

    test_env = ContinuousIrregularFLPEnv(
        n_facilities=len(station_refs),
        excel_data=shared_data,
        render_mode="rgb_array",
        fixed_scenario=scenario_name,
        max_steps=max_steps
    )
    test_env = PPOCompatibleEnv(test_env)

    results = []

    for episode in range(num_episodes):
        obs, info = test_env.reset()
        episode_reward = 0
        episode_cost = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            episode_reward += reward
            episode_cost = info.get('cost', 0)
            done = terminated or truncated

        final_layout = test_env.env.current_layout.copy()
        results.append({
            'episode': episode + 1,
            'layout': final_layout,
            'final_cost': episode_cost,
            'total_reward': episode_reward
        })

        if (episode + 1) % 20 == 0:
            print(f"[TEST] Episode {episode + 1}/{num_episodes} completed.")
            sys.stdout.flush()

    test_env.close()

    # Sort results by final cost (ascending)
    results.sort(key=lambda x: x['final_cost'])
    return results


# Test on regular flow scenario
# Run 160 episodes
regular_flow_results = test_on_scenario("regular", num_episodes=160, max_steps=200)

# Statistical summary
costs = [r['final_cost'] for r in regular_flow_results]
rewards = [r['total_reward'] for r in regular_flow_results]

print("\n📊 Statistical Summary (Regular Flow Scenario):")
print(f"Costs (in millions): Mean = {np.mean(costs)/1_000_000:.2f}, Median = {np.median(costs)/1_000_000:.2f}, "
      f"Std = {np.std(costs)/1_000_000:.2f}, Min = {np.min(costs)/1_000_000:.2f}, Max = {np.max(costs)/1_000_000:.2f}")
print(f"Rewards: Mean = {np.mean(rewards):.4f}, Median = {np.median(rewards):.4f}, "
      f"Std = {np.std(rewards):.4f}, Min = {np.min(rewards):.4f}, Max = {np.max(rewards):.4f}")


# PHASE 3: VISUALIZATION IN HUMAN MODE
print("\n" + "=" * 60)
print("PHASE 3: RENDERING LAYOUTS IN HUMAN MODE")
print("=" * 60)


def render_layout_human(scenario_name, layout, final_cost, total_reward, env):
    """
    Render a single layout using human mode visualization.
    """
    print(f"\n[RENDER] Displaying layout for '{scenario_name}' scenario...")
    print(f"[RENDER] Final Cost: {final_cost:.2f}, Total Reward: {total_reward:.4f}")
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
    colors = plt.cm.tab10(np.linspace(0, 1, env.n_facilities))# type: ignore
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
        f"Facility Layout - {scenario_name.upper()} Flow Scenario\n"
        f"Cost: {final_cost / 1_000_000:.2f}M | Reward: {total_reward:.4f}",
        fontsize=14, fontweight='bold'
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# Render regular flow results
print("\n[RENDER] REGULAR FLOW SCENARIO RESULTS:")
stations_df = shared_data["Stations"]
station_refs = [s for s in stations_df["Number"] if str(s).startswith("S")]
regular_env = ContinuousIrregularFLPEnv(
    n_facilities=len(station_refs),
    excel_data=shared_data,
    render_mode="human",
    fixed_scenario="regular"
)
top5 = sorted(regular_flow_results, key=lambda x: x['final_cost'])[:5]
for result in top5:
    render_layout_human(
        "regular",
        result['layout'],
        result['final_cost'],
        result['total_reward'],
        regular_env
    )

# SUMMARY
print("\n📊 Summary:")
print(f"  ✓ Training completed on 'regular' flow scenario")
print(f"  ✓ Tested on 'regular' flow scenario: {len(regular_flow_results)} episodes")
