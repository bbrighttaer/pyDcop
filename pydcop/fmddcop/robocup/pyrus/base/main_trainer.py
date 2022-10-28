#!/usr/bin/python3
from pydcop.fmddcop.robocup.pyrus.base.sample_trainer import SampleTrainer
from pydcop.fmddcop.robocup.pyrus.pyruslib.player.basic_client import BasicClient

import sys


def main(team_name="Pyrus", goalie=False):
    agent = SampleTrainer()
    client = BasicClient()
    agent.init(client)

    client.run(agent)
    # agent.run(team_name, goalie)


if __name__ == "__main__":
    goalie = False
    if len(sys.argv) > 1 and sys.argv[1] == "g":
        goalie = True
    main(goalie=goalie)
