import numpy as np
import pandas as pd
import random

class NSGAII:
    def __init__(self, pop_size, num_features, targets_list, fraction_indices,target_directions,
                 crossover_prob=0.85, mutation_prob=0.15, eta_crossover=20, eta_mutation=30):
        self.pop_size = pop_size
        self.num_features = num_features
        self.num_objectives = len(targets_list)
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.eta_crossover = eta_crossover
        self.eta_mutation = eta_mutation
        self.fraction_indices = fraction_indices
        self.target_directions = np.array(target_directions)

        
    def initialize_population(self):
        population = np.zeros((self.pop_size, self.num_features))
        for j in range(self.num_features):
            population[:, j] = np.random.permutation(np.linspace(0, 1, self.pop_size))
        for i in range(self.pop_size):
            fractions = np.random.dirichlet([1] * len(self.fraction_indices))
            population[i, self.fraction_indices] = fractions
        return population

    def dominates(self, a, b):

        a_better = False
        for i in range(len(a)):
            if self.target_directions[i] == 1:  # 最大化
                if a[i] < b[i]:
                    return False
                if a[i] > b[i]:
                    a_better = True
            else:  # 最小化
                if a[i] > b[i]:
                    return False
                if a[i] < b[i]:
                    a_better = True
        return a_better


    def fast_non_dominated_sort(self, population, objectives):
        N = len(population)
        fronts = [[]]
        domination_counts = np.zeros(N, dtype=int)
        dominated_solutions = [[] for _ in range(N)]

        for i in range(N):
            obj_i = objectives[i]
            for j in range(i + 1, N):
                obj_j = objectives[j]
                if self.dominates(obj_i, obj_j):
                    dominated_solutions[i].append(j)
                    domination_counts[j] += 1
                elif self.dominates(obj_j, obj_i):
                    dominated_solutions[j].append(i)
                    domination_counts[i] += 1
            if domination_counts[i] == 0:
                fronts[0].append(i)

        while fronts[-1]:
            next_front = []
            for i in fronts[-1]:
                for j in dominated_solutions[i]:
                    domination_counts[j] -= 1
                    if domination_counts[j] == 0:
                        next_front.append(j)
            fronts.append(next_front)

        fronts.pop()
        return fronts


    def crowding_distance(self, front_indices, combined_obj):
        if len(front_indices) == 0:
            return np.zeros(0)

        front_obj = combined_obj[front_indices]
        distances = np.zeros(len(front_indices))

        for m in range(self.num_objectives):
            values = front_obj[:, m]
            sorted_idx  = np.argsort(values)
            min_val, max_val = front_obj[sorted_idx[0], m], front_obj[sorted_idx[-1], m]
            distances[sorted_idx[0]] = distances[sorted_idx[-1]] = float('inf')

            if max_val == min_val:
                continue

            norm = max_val - min_val
            distances[sorted_idx[1:-1]] += (
                                            values[sorted_idx[2:]] - values[sorted_idx[:-2]]
                                        ) / norm
        return distances

    def sbx_crossover(self, parent1, parent2):
        if random.random() > self.crossover_prob:
            child1, child2 = parent1.copy(), parent2.copy()
        else:
            child1 = np.empty_like(parent1)
            child2 = np.empty_like(parent2)

            u = np.random.rand(len(parent1))

            beta = np.where(
            u <= 0.5,
            (2 * u) ** (1 / (self.eta_crossover + 1)),
            (1 / (2 * (1 - u))) ** (1 / (self.eta_crossover + 1))
            )

            child1 = 0.5 * ((1 + beta) * parent1 + (1 - beta) * parent2)
            child2 = 0.5 * ((1 - beta) * parent1 + (1 + beta) * parent2)

            child1 = np.clip(child1, 0, 1)
            child2 = np.clip(child2, 0, 1)

        for child in [child1, child2]:
            fractions = child[self.fraction_indices]
            total = fractions.sum()
            if total > 0:
                child[self.fraction_indices] = fractions / total
            else:
                child[self.fraction_indices] = np.random.dirichlet([1] * len(self.fraction_indices))
            
        return child1, child2

    def polynomial_mutation(self, individual):
        mutated = individual.copy()
        prob = self.mutation_prob
        u = np.random.rand(self.num_features)
        mutate_mask = u < prob
        rand_vals = np.random.rand(self.num_features)
        delta_q = np.where(
                rand_vals < 0.5,
                (2 * rand_vals) ** (1 / (self.eta_mutation + 1)) - 1,
                1 - (2 * (1 - rand_vals)) ** (1 / (self.eta_mutation + 1))
                )
        mutated[mutate_mask] += delta_q[mutate_mask]

        noise_mask = np.random.rand(self.num_features) < 0.2
        mutated[noise_mask] += np.random.normal(0, 0.05, size=self.num_features)[noise_mask]

        mutated = np.clip(mutated, 0, 1)

        fractions = mutated[self.fraction_indices]
        total = fractions.sum()
        if total > 0:
            mutated[self.fraction_indices] = fractions / total
        else:
            mutated[self.fraction_indices] = np.random.dirichlet([1] * len(self.fraction_indices))
        mutated[self.fraction_indices] = np.clip(mutated[self.fraction_indices], 0, 1)

        return mutated

    def evolve(self, parent_pop, parent_obj, offspring_pop, offspring_obj):
        combined_pop = np.vstack([parent_pop, offspring_pop])
        combined_obj = np.vstack([parent_obj, offspring_obj])

        fronts = self.fast_non_dominated_sort(combined_pop, combined_obj)

        new_pop = np.empty((self.pop_size, combined_pop.shape[1]))
        new_obj = np.empty((self.pop_size, combined_obj.shape[1]))
        fill = 0

        for front in fronts:
            if len(front) == 0:
                continue
            front_size = len(front)
            if fill + front_size <= self.pop_size:
                new_pop[fill:fill + front_size] = combined_pop[front]
                new_obj[fill:fill + front_size] = combined_obj[front]
                fill += front_size
            else:
                remaining = self.pop_size - fill
                last_front = np.array(front, dtype=int)
                crowding = self.crowding_distance(last_front, combined_obj)
                selected = last_front[np.argsort(-crowding)[:remaining]]
                new_pop[fill:] = combined_pop[selected]
                new_obj[fill:] = combined_obj[selected]
                break

        return new_pop, new_obj