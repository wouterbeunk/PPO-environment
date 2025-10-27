import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PPO_data_loader import data_loader

class ContinuousIrregularFLPEnv(gym.Env):
    # two render_modes are set: human if you want to visualize it, rgb_array for computational processing
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self,
        n_facilities,
        max_steps: int = 200,
        render_mode: str = "human",
        seed: int = 42):

        super().__init__()

        np.random.seed(seed)

        self.max_steps = max_steps
        self.render_mode = render_mode

        # Defining main placement areas (x, y, width, height (x and y coordinates are bottom left corner))
        self.main_area = (0, 0, 29.0, 28.82)
        self.top_right_area = (12.04, 29.1, 17.06, 10.88)
        self.mid_right_area = (29.1, 18.5, 7.9, 4.2)

        # total area of the entire field (including infeasible areas)
        self.total_width = 37.0
        self.total_height = 39.98

        # setting the restricted zones (x, y, width, height (x and y coordinates are bottom left corner))
        self.restricted_zones = [
            (0, 4.75, 0.5, 0.5),
            (6.18,4.75, 0.5, 0.5),
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

        # importing excel data with use of custom-made PPO_data_loader
        filepath = "C:\\Users\\beunk\Downloads\\Bij-producten meta data.xlsx"
        data_import = data_loader(filepath)
        extracted_data = data_import.excel_file_loading()

        # setting the correct dataframes
        stations_df = extracted_data["Stations"]
        flow_df = extracted_data["Test flow matrix"]

        # creating variables to link the machines of both sheets with each other
        station_refs = list(stations_df["Number"])
        flow_col_refs = list(flow_df.columns)
        flow_row_refs = list(flow_df.index)

        # reindexing the flow matrix to guarantee they correctly link
        flow_df = flow_df.loc[station_refs, station_refs]

        self.flow_col_refs = flow_col_refs
        self.flow_row_refs = flow_row_refs
        self.station_references = station_refs
        self.flow_matrix = flow_df

        # creating variables with the values from the excel file
        self.machine_dimensions = stations_df[["Width", "Height"]].values.tolist()
        self.n_facilities = len(station_refs)
        self.flow_matrix = flow_df.values

        # creating the action space dictionary, because facilityID is discrete and placement should be continuous
        # both x and y coordinates of the facilities can be moved in a continuous action space ranging from -1 until 1
        self.action_space = spaces.Dict({
            'facility_id': spaces.Discrete(n_facilities),
            'move': spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        })

        # setting the total possible area where the agent can look (so from 0, 0 until the max width and height)
        self.observation_space = spaces.Box(
            low=np.float32(0.0),
            high=np.float32(max(self.total_width, self.total_height)),
            shape=(self.n_facilities * 2,),
            dtype=np.float32
        )

        # Initializing variables
        self.current_layout = None
        self.step_count = 0
        self.previous_cost = float('inf')


    # check whether facilities can be placed within the separate areas, so they will not be placed outside
    def _is_in_irregular_area(self, x, y, w, h):
        def inside_main(px, py):
            return 0 <= px <= 29.0 and 0 <= py <= 28.82

        def inside_top_right(px, py):
            return 12.04 <= px <= 29.1 and 29.1 <= py <= 39.98

        def inside_mid_right(px, py):
            return 29.1 <= px <= 37 and 18.5 <= py <= 22.7

        corners = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]
        for (px, py) in corners:
            if not (inside_main(px, py) or inside_top_right(px, py) or inside_mid_right(px, py)):
                return False
        return True

    # creating a function for restricted zones
    def _is_in_restricted_zone(self, x, y, w, h):
        for (rx, ry, rw, rh) in self.restricted_zones:
            if not (x + w <= rx or rx + rw <= x or y + h <= ry or ry + rh <= y):
                return True
        return False

    # combining previous function and determining whether the placement is valid or not
    def _is_valid_placement(self, x, y, w, h, facility_id, layout=None):
        if not self._is_in_irregular_area(x, y, w, h):
            return False
        if self._is_in_restricted_zone(x, y, w, h):
            return False

        #check for overlap in placing facilities
        if layout is None:
            layout = self.current_layout
        for other_id, (ox, oy, ow, oh) in layout.items():
            if other_id == facility_id:
                continue
            if not (x + w <= ox or ox + ow <= x or y + h <= oy or oy + oh <= y):
                return False
        return True

    def _generate_random_layout(self):
        layout = {}
        max_retries = 200

        # create the faciliteis
        for i in range(self.n_facilities):
            width, height = self.machine_dimensions[i]
            placed = False
        # try to place the facilities
            for _ in range(max_retries):
                x = np.random.uniform(0, self.total_width - width)
                y = np.random.uniform(0, self.total_height - height)
        # check whether facility i can be placed
                if self._is_valid_placement(x, y, width, height, i, layout):
                    layout[i] = (x, y, width, height)
                    placed = True
                    break
        # all facilities need to be placed otherwise model will break
            if not placed:
                raise RuntimeError(
                    f"Failed to place facility {i} (size {width}x{height}) after "
                    f"{max_retries} attempts. Check if layout space is sufficient."
                )
        return layout

    # reset the environment at the beginning of placing facilities
    def reset(self, *, seed=None, options=None):
        # reset feature from gymnasium library taken
        super().reset(seed=seed)
        # set everything to default after a reset
        self.step_count = 0
        self.previous_cost = float('inf')
        self.current_layout = self._generate_random_layout()
        obs = self._get_observation()
        info = {}
        return obs, info

    # when the agent takes an action
    def step(self, action):
        self.step_count += 1
        facility_id = action['facility_id']
        # np.clip makes sure that the movement of facility stays in the defined range
        dx, dy = np.clip(action['move'], -1.0, 1.0)
        x, y, w, h = self.current_layout[facility_id]
        new_x = np.clip(x + dx, 0, self.total_width - w)
        new_y = np.clip(y + dy, 0, self.total_height - h)

        # makes a copy of the layout and places facilities + checks them
        temp_layout = self.current_layout.copy()
        temp_layout[facility_id] = (new_x, new_y, w, h)
        if self._is_valid_placement(new_x, new_y, w, h, facility_id, temp_layout):
            self.current_layout = temp_layout

        # reward function for the agent
        cost = self._calculate_material_handling_cost()
        # if new cost < old cost = pos reward / if new cost > old cost = negative penalty reward
        reward = (self.previous_cost - cost) / (abs(self.previous_cost) + 1e-6)
        # reward cannot be too large to maintain stable learning
        reward = np.clip(reward, -1, 1)

        if not self._check_valid_layout():
            reward -= 0.5

        self.previous_cost = cost

        terminated = self.step_count >= self.max_steps
        # truncated = if action is unsuccesful (not coded yet)
        truncated = False
        return self._get_observation(), reward, terminated, truncated, {'cost': cost}

    def _calculate_material_handling_cost(self):
        total = 0
        # if a facility is not placed, it skips it
        for i in range(self.n_facilities):
            if i not in self.current_layout:
                continue
            # if a facility is not placed, it skips it
            # j always needs to be larger than i, so it doesn't calculate material flow costs to itself
            for j in range(i + 1, self.n_facilities):
                if j not in self.current_layout:
                    continue
                flow = self.flow_matrix[i, j]
                xi = self.current_layout[i][0]
                yi = self.current_layout[i][1]
                xj = self.current_layout[j][0]
                yj = self.current_layout[j][1]
                euclidean_distance = np.sqrt((xj - xi)**2 + (yj - yi)**2)
                total += flow * euclidean_distance
        return total

    # checks the entire layout based on valid placements
    def _check_valid_layout(self):
        for fid, (x, y, w, h) in self.current_layout.items():
            if not self._is_valid_placement(x, y, w, h, fid):
                return False
        return True

    def _get_observation(self):
        obs = []
        for fid in range(self.n_facilities):
            if fid in self.current_layout:
                x, y, _, _ = self.current_layout[fid]
            else:
                x, y = 0, 0
            obs.extend([x / self.total_width, y / self.total_height])
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

        # Draw facilities
        colors = plt.cm.tab10(np.linspace(0, 1, self.n_facilities))
        for fid, (x, y, w, h) in self.current_layout.items():
            rect = Rectangle((x, y), w, h, linewidth=1.5,
                             edgecolor='black', facecolor=colors[fid % 10], alpha=0.8)
            ax.add_patch(rect)
            ax.text(x + w / 2, y + h / 2, str(fid),
                    ha='center', va='center', fontsize=10, color='black', fontweight='bold')

        ax.set_xlim(-1, self.total_width + 1)
        ax.set_ylim(-1, self.total_height + 1)
        ax.set_aspect('equal')
        ax.set_title(f"Irregular Layout - Step {self.step_count}")

        if self.render_mode == "rgb_array":
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            plt.close(fig)
            return img[:, :, :3]

        elif self.render_mode == "human":
            plt.show() #block=False
            #plt.pause(0.001)
            return None


# Test environment visually


if __name__ == "__main__":
    env = ContinuousIrregularFLPEnv(n_facilities=17, render_mode="rgb_array")
    obs = env.reset()
    frame = env.render()
    total_cost = env._calculate_material_handling_cost()
    print("the total cost for this layout is: ", total_cost)
    for _ in range(0):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated:
            break
