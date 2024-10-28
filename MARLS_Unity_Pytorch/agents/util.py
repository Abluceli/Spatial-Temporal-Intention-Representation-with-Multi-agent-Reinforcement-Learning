import math


class DecayedValue:
    def __init__(
            self,
            scheduletype: str,
            initial_value: float,
            min_value: float,
            max_step: int
    ):
        """
        Object that represnets value of a parameter that should be decayed, assuming it is a function of
        global_step.
        :param scheduletype: Type of learning rate schedule.
        :param initial_value: Initial value before decay.
        :param min_value: Decay value to this value by max_step.
        :param max_step: The final step count where the return value should equal min_value.
        :param global_step: The current step count.
        :return: The value.
        """
        self.schedule = scheduletype
        self.initial_value = initial_value
        self.min_value = min_value
        self.max_step = max_step

    def get_value(self, global_step: int) -> float:
        """
        Get the value at a given global step.
        :param global_step: Step count.
        :returns: Decayed value at this global step.
        """
        if self.schedule == "CONSTANT":
            return self.initial_value
        elif self.schedule == "LINEAR":
            return self.polynomial_decay(self.initial_value, self.min_value, self.max_step, global_step)
        else:
            raise print(f"The schedule {self.schedule} is invalid.")

    def polynomial_decay(self,
                         initial_value: float,
                         min_value: float,
                         max_step: int,
                         global_step: int,
                         power: float = 1.0,
                         ) -> float:
        """
        Get a decayed value based on a polynomial schedule, with respect to the current global step.
        :param initial_value: Initial value before decay.
        :param min_value: Decay value to this value by max_step.
        :param max_step: The final step count where the return value should equal min_value.
        :param global_step: The current step count.
        :param power: Power of polynomial decay. 1.0 (default) is a linear decay.
        :return: The current decayed value.
        """
        global_step = min(global_step, max_step)
        decayed_value = (initial_value - min_value) * (
                1 - float(global_step) / max_step
        ) ** (power) + min_value
        return decayed_value


class EpsilonScheduler():
    def __init__(self, eps_start, eps_final, eps_decay):
        """A scheduler for epsilon-greedy strategy.
        :param eps_start: starting value of epsilon, default 1. as purely random policy
        :type eps_start: float
        :param eps_final: final value of epsilon
        :type eps_final: float
        :param eps_decay: number of timesteps from eps_start to eps_final
        :type eps_decay: int
        """
        self.eps_start = eps_start
        self.eps_final = eps_final
        self.eps_decay = eps_decay
        self.epsilon = self.eps_start
        self.ini_frame_idx = 0
        self.current_frame_idx = 0

    def reset(self, ):
        """ Reset the scheduler """
        self.ini_frame_idx = self.current_frame_idx

    def step(self, frame_idx):
        self.current_frame_idx = frame_idx
        delta_frame_idx = self.current_frame_idx - self.ini_frame_idx
        self.epsilon = self.eps_final + (self.eps_start - self.eps_final) * math.exp(
            -1. * delta_frame_idx / self.eps_decay)

    def get_epsilon(self):
        return self.epsilon