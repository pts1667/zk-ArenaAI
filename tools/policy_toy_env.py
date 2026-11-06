"""The toy as an environment for tools/policy_train.py: many games, some seats played by the policy, some by scripts."""
from __future__ import annotations

import numpy as np

import policy_toy as toy


class ToyEnv:
    eval_opponents = ("nearest", "random")

    def __init__(self, games: int, seed: int, modes: list[str], decide_every: int = 1):
        self.modes = modes
        self.decide_every = decide_every    # the policy's rate; scripts still act on every slot, so a slower policy gets no slower opponent
        self.rng = np.random.default_rng(seed)
        self.games: list[toy.ToyGame | None] = [None] * games
        self.mode = [modes[i % len(modes)] for i in range(games)]
        self.done: list[tuple[int, str, float]] = []
        self.pending: dict[int, list] = {}
        self.max_regions = toy.plain().regions
        self.seat_count = games * 2      # seat s plays side s % 2 of game s // 2
        self.last_raw = np.zeros(games)
        self._replace: list[int] = []

    def begin(self) -> None:
        for g in range(len(self.games)):
            self._new_game(g)

    def _new_game(self, g: int) -> None:
        self.mode[g] = self.modes[int(self.rng.integers(len(self.modes)))]
        self.games[g] = toy.ToyGame(self.rng)
        self.last_raw[g] = 0.0

    def _is_policy(self, seat: int) -> bool:
        return seat % 2 == 0 or self.mode[seat // 2] == "mirror"

    def policy_seats(self) -> list[int]:
        """Every seat that might ever be the policy's."""
        return list(range(self.seat_count))

    def policy_seats_to_act(self) -> list[int]:
        return [s for s in range(self.seat_count) if self._is_policy(s)]

    def bundles(self) -> list:
        return [toy.plain()]

    def bundle_of(self, seat: int):
        return toy.plain()

    def frame(self, seat: int):
        return self.games[seat // 2].frame_for(seat % 2)

    def give_orders(self, seat: int, orders: list) -> None:
        self.pending[seat] = orders

    def step_scripted_only(self) -> None:
        self.step(0.0)

    def step(self, progress: float) -> None:
        self.done = []
        for g, game in enumerate(self.games):
            hit = np.zeros_like(game.damaged)
            for slot in range(self.decide_every):
                for side in (0, 1):
                    seat = g * 2 + side
                    if slot == 0 and seat in self.pending:
                        game.apply(side, self.pending.pop(seat))
                    elif not self._is_policy(seat):
                        game.apply(side, toy.SCRIPTS[self.mode[g]](game, side, self.rng))
                game.step()
                hit |= game.damaged
                if game.winner() is not None:
                    break
            game.damaged = hit      # since the policy last looked
            raw = game.raw_value() - game.start_raw
            game.delta = raw - self.last_raw[g]
            self.last_raw[g] = raw
            winner = game.winner()
            game.result = 0.0
            if winner is not None:
                game.result = {0: 1.0, 1: -1.0, 2: 0.0}[winner]
                game.ended = True
                for side in (0, 1):
                    seat = g * 2 + side
                    if self._is_policy(seat):
                        z = {0: 1.0, 1: 0.0, 2: 0.5}[winner] if side == 0 else {0: 0.0, 1: 1.0, 2: 0.5}[winner]
                        self.done.append((seat, self.mode[g], z))
            else:
                game.ended = False
        self._replace = [g for g, game in enumerate(self.games) if game.ended]

    def reward(self, seat: int, beta: float) -> float:
        game = self.games[seat // 2]
        sign = 1.0 if seat % 2 == 0 else -1.0
        return sign * ((1.0 - beta) * game.delta + beta * game.result)

    def finished_seats(self) -> list[tuple[int, str, float]]:
        """Seats whose game just ended.  The call also starts those games afresh, so read the reward first."""
        out = self.done
        for g in self._replace:
            self._new_game(g)
        self._replace = []
        self.done = []
        return out
