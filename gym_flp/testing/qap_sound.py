import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from sklearn.manifold import MDS
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


class FacilityLayoutEnv(gym.Env):
    def __init__(self, num_machines, flow_matrix, sound_levels, location_distances=None, grid_size=(10, 10)):
        super(FacilityLayoutEnv, self).__init__()
        self.num_machines = num_machines
        self.flow_matrix = flow_matrix
        self.sound_levels = sound_levels
        self.grid_size = grid_size  # Size of the factory grid

        if location_distances is None:
            # If location distances are not provided, generate random machine locations and calculate distances
            self.machine_locations = self._generate_random_machine_locations()
            self.location_distances = self._calculate_location_distances()
        else:
            self.location_distances = location_distances
            self.machine_locations = self._infer_locations_from_distances(location_distances)

        print(self.machine_locations)
        # Define action and observation space
        self.action_space = spaces.Discrete(num_machines)  # Each action corresponds to a location
        self.observation_space = spaces.MultiDiscrete(
            [num_machines + 1] * num_machines)  # Machine to location assignments

        self.grid_positions = self._generate_grid_positions()
        self.reset()

    def reset(self, *, seed = None, options = None):
        # Initialize queue of machines to be assigned
        self.machine_queue = list(range(self.num_machines))
        self.current_assignments = [self.num_machines] * self.num_machines  # Use num_machines to indicate unassigned
        self.used_locations = set()
        self.current_step = 0
        self.initial_cost = self._calculate_material_handling_cost()  # Initial cost for normalization

        return self._get_obs(), {}

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"

        if action in self.used_locations:
            # Assign a negative penalty if the location has been used before
            reward = -1000
            done = True
            final_cost = self._calculate_material_handling_cost()
        else:
            machine = self.machine_queue[self.current_step]
            self.current_assignments[machine] = action
            self.used_locations.add(action)
            self.current_step += 1

            done = self.current_step >= self.num_machines
            if done:
                final_cost = self._calculate_material_handling_cost()
                normalized_cost = final_cost / self.initial_cost if self.initial_cost != 0 else final_cost
                avg_sound_level = self._calculate_average_sound_level()
                reward = -normalized_cost - avg_sound_level  # Negative reward based on normalized cost and average sound level
            else:
                reward = 0  # No reward until all machines are assigned

        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        return np.array(self.current_assignments)

    def _calculate_material_handling_cost(self):
        cost = 0
        for i in range(self.num_machines):
            for j in range(i + 1, self.num_machines):
                if self.current_assignments[i] < self.num_machines and self.current_assignments[j] < self.num_machines:
                    loc_i = self.current_assignments[i]
                    loc_j = self.current_assignments[j]
                    distance = self.location_distances[loc_i][loc_j]
                    flow = self.flow_matrix[i][j]
                    cost += distance * flow
        return cost

    def _calculate_average_sound_level(self):
        total_sound_power = 0
        for i in range(self.num_machines):
            if self.current_assignments[i] < self.num_machines:
                loc = self.current_assignments[i]
                machine_x, machine_y = self.machine_locations[loc]
                distance = self._average_distance_to_all(machine_x, machine_y)
                attenuation_factor = 1 / (distance ** 2 + 1)  # Simple attenuation model
                total_sound_power += 10 ** (self.sound_levels[i] / 10) * attenuation_factor

        avg_sound_level = 10 * np.log10(total_sound_power) if total_sound_power > 0 else 0
        return avg_sound_level

    def _average_distance_to_all(self, x, y):
        total_distance = 0
        count = 0
        for i in range(self.num_machines):
            if self.current_assignments[i] < self.num_machines:
                assigned_x, assigned_y = self.machine_locations[self.current_assignments[i]]
                total_distance += np.sqrt((x - assigned_x) ** 2 + (y - assigned_y) ** 2)
                count += 1
        return total_distance / count if count > 0 else 1  # Avoid division by zero

    def _calculate_location_distances(self):
        distances = np.zeros((self.num_machines, self.num_machines))
        for i in range(self.num_machines):
            for j in range(self.num_machines):
                if i != j:
                    x1, y1 = self.machine_locations[i]
                    x2, y2 = self.machine_locations[j]
                    distances[i, j] = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        return distances

    def _generate_grid_positions(self):
        positions = []
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                positions.append((i, j))
        return positions

    def _generate_random_machine_locations(self):
        return [(random.randint(0, self.grid_size[0] - 1), random.randint(0, self.grid_size[1] - 1)) for _ in
                range(self.num_machines)]

    def _infer_locations_from_distances(self, distance_matrix):
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=0)
        inferred_locations = mds.fit_transform(distance_matrix)
        # Scale and translate the coordinates to fit within the grid size
        inferred_locations -= inferred_locations.min(axis=0)
        inferred_locations /= inferred_locations.max(axis=0)
        inferred_locations *= (np.array(self.grid_size) - 1)
        return [(int(x), int(y)) for x, y in inferred_locations]

    def _calculate_sound_at_point(self, x, y):
        total_sound_power = 0
        for i in range(self.num_machines):
            if self.current_assignments[i] < self.num_machines:
                loc = self.current_assignments[i]
                machine_x, machine_y = self.machine_locations[loc]
                distance = np.sqrt((x - machine_x) ** 2 + (y - machine_y) ** 2)
                attenuation_factor = 1 / (distance ** 2 + 1)  # Simple attenuation model
                total_sound_power += 10 ** (self.sound_levels[i] / 10) * attenuation_factor

        sound_level = 10 * np.log10(total_sound_power) if total_sound_power > 0 else 0
        return sound_level

    def render(self, mode='human'):
        # Render the facility layout as a heatmap
        heatmap = np.zeros(self.grid_size)
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                heatmap[x, y] = self._calculate_sound_at_point(x, y)

        plt.figure(figsize=(8, 6))
        plt.imshow(heatmap, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Sound Pressure Level (dB)')

        # Plot machine locations
        for loc in self.machine_locations:
            plt.scatter(loc[1], loc[0], color='blue', s=100)  # Note: scatter uses (x, y) where x is column, y is row

        plt.title("Sound Pressure Level Heatmap")
        plt.show()

    def close(self):
        pass


# Example usage
if __name__ == "__main__":
    num_machines = 4

    # Generate random sound pressure levels for machines
    sound_levels = [random.randint(70, 100) for _ in range(num_machines)]

    # Generate random symmetrical distance matrix
    location_distances = np.random.randint(1, 10, size=(num_machines, num_machines))
    location_distances = (location_distances + location_distances.T) / 2
    np.fill_diagonal(location_distances, 0)

    location_distances = np.array(([[0, 22, 53, 53],
                                    [22, 0, 40, 62],
                                    [53, 40, 0, 55],
                                    [53, 62, 55, 0]]))


    # Ensure flow matrix is symmetrical
    flow_matrix = np.random.randint(1, 10, size=(num_machines, num_machines))
    flow_matrix = (flow_matrix + flow_matrix.T) // 2

    flow_matrix = np.array(([[0, 3, 0, 2],
                            [3, 0, 0, 1],
                            [0, 0, 0, 4],
                            [2, 1, 4, 0]]))

    env = FacilityLayoutEnv(num_machines, flow_matrix, sound_levels, location_distances, grid_size=(150, 150))

    # Use make_vec_env to handle multiple environments (for parallel processing)
    vec_env = make_vec_env(lambda: env, n_envs=1)

    # Define the PPO model
    model = PPO('MlpPolicy', vec_env, verbose=1)

    # Train the model
    model.learn(total_timesteps=100000)

    # Test the trained model
    obs = vec_env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = vec_env.step(action)
        print(action, obs)
        env.render()

    print(f"Final Reward (negative cost): {rewards}")
    print(env.machine_locations)

