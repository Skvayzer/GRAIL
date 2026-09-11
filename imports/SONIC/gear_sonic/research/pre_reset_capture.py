"""Instance-local tap after reward computation, before Isaac auto-reset.

No reordered simulation operations, added physics steps, command advancement,
observation-manager calls or RNG consumption. Callback must be read-only.
"""
class PreResetCapture:
    def __init__(self, env, callback):
        self.env, self.callback = env, callback
        self.pending = None
        self.calls = 0

    def __enter__(self):
        manager = self.env.reward_manager
        if "compute" in vars(manager):
            raise ValueError("Reward computation already has an instance override")
        self.original = manager.compute

        def compute(*args, **kwargs):
            if self.pending is not None:
                raise ValueError("Previous pre-reset sample was not consumed")
            reward = self.original(*args, **kwargs)
            self.pending = self.callback(reward)
            if self.pending is None:
                raise ValueError("Pre-reset callback must return an explicit sample")
            self.calls += 1
            return reward

        manager.compute = compute
        self.installed = compute
        return self

    def take(self):
        if self.pending is None:
            raise ValueError("Missing pre-reset transition capture")
        result, self.pending = self.pending, None
        return result

    def __exit__(self, kind, error, traceback):
        if self.env.reward_manager.compute is not self.installed:
            raise ValueError("Another owner changed reward computation during capture")
        del self.env.reward_manager.compute
        return False
