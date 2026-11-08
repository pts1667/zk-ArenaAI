"""The policy host (bots/zkbot/host.py): what it keeps from observations is what a raw frame is read from."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "bots"))

from zkbot import units  # noqa: E402
from zkbot.host import Host  # noqa: E402
from zkbot.mapdata import MetalSpot  # noqa: E402
from zkbot.policy import rawframe as rf  # noqa: E402


class Client:
    frame = 0

    def __init__(self):
        self.sent, self.settings = [], []

    def send(self, msg):
        self.sent.append(msg)

    def set_settings(self, settings):
        self.settings.append(settings)

    def query(self, q, timeout=10.0):
        return {"floatValue": 1.0}


def _unit(name, x, z, **extra):
    return {"defId": units.id_of(name), "pos": [x, 0.0, z], "vel": [0.0, 0.0, 0.0], "health": 100.0, "maxHealth": 100.0, "buildProgress": 1.0, **extra}


def _host():
    host = Host(Client(), 0, lambda _m: None)
    host.on_bootstrap("game_info", {"allyTeamId": 0, "mapWidthElmos": 4096, "mapHeightElmos": 4096})
    host.map.spots = [MetalSpot(500.0, 500.0, 2.0, 0), MetalSpot(3000.0, 3000.0, 2.0, 1), MetalSpot(2000.0, 2000.0, 2.0, 2)]
    return host


def test_it_chooses_a_commander_before_the_game_and_sends_nothing_after():
    host = _host()
    host.on_startup({"frame": -1})
    assert host.client.sent[0] == {"type": "set_commander", "name": "dyntrainer_support_base"}
    host.on_frame({"frame": 30, "units": {"1": _unit("dyntrainer_support_base", 400.0, 450.0)}})
    assert host.take_commands() == []
    assert host.commander_id == 1 and host.base_pos == (400.0, 450.0)


def test_what_it_keeps_reads_as_a_raw_frame():
    host = _host()
    host.on_frame({"frame": 30, "economy": {"Metal": {"current": 70.0, "income": 4.0, "storage": 10500.0}},
                   "units": {"1": _unit("dyntrainer_support_base", 400.0, 450.0), "2": _unit("staticmex", 500.0, 500.0)},
                   "enemies": {"20": _unit("staticmex", 3000.0, 3000.0, allyTeam=1), "21": _unit("cloakraid", 2500.0, 2500.0, allyTeam=1)},
                   "radarBlips": {"30": {"allyTeam": 1, "pos": [2600.0, 0.0, 2600.0], "vel": [2.0, 0.0, 0.0]}}})
    for frame in range(31, 31 + 60, 6):         # a blip's speed is the least it held over two seconds
        host.on_frame({"frame": frame, "units": {"1": _unit("dyntrainer_support_base", 400.0, 450.0)},
                       "radarBlips": {"30": {"allyTeam": 1, "pos": [2600.0, 0.0, 2600.0], "vel": [2.0, 0.0, 0.0]}}})
    host.on_frame({"frame": 105, "economy": {"Metal": {"current": 70.0, "income": 4.0, "storage": 10500.0}},
                   "units": {"1": _unit("dyntrainer_support_base", 400.0, 450.0), "2": _unit("staticmex", 500.0, 500.0)}})
    assert [s.status for s in host.map.spots] == ["ours", "enemy", "free"]
    f = rf.from_host(host, {"frame": 105, "units": {"1": _unit("dyntrainer_support_base", 400.0, 450.0)}}, rf.LiveMemory())
    by_uid = {int(r["uid"]): r for r in f.units}
    assert by_uid[20]["knowledge"] == rf.KNOW_MEMORY and 21 not in by_uid        # the extractor is remembered, the raider isn't
    assert f.glob[rf.G["metal_storage"]] == 500.0 and f.glob[rf.G["commander_uid"]] == 1
    assert f.glob[rf.G["frames_since_contact"]] == 105 - 85
    assert host.blip_tracks[30]["smax"] == 60.0


def test_a_remembered_building_is_forgotten_once_its_place_is_seen_empty():
    host = _host()
    host.on_frame({"frame": 30, "units": {"1": _unit("dyntrainer_support_base", 400.0, 450.0)},
                   "enemies": {"20": _unit("staticmex", 3000.0, 3000.0, allyTeam=1)}})
    assert 20 in host.enemy_structures
    host.on_frame({"frame": 36, "units": {"1": _unit("dyntrainer_support_base", 400.0, 450.0), "2": _unit("cloakraid", 3010.0, 3000.0)}})
    assert 20 not in host.enemy_structures
