
class MobileSensor:

    def __init__(self, player_id):
        super().__init__()
        self.player_id = player_id
        self.client = None
        self.current_position = None
        self.credibility = 5
        self.sensing_range = 1
        self.mobility_range = 2
        self.connectivity_range = 3


class Target:

    def __init__(self, target_id, position, cov_req):
        self.target_id = target_id
        self.current_position = position
        self.coverage_requirement = cov_req
        self.is_active = True


class GridCell:

    def __init__(self, cell_id):
        self.cell_id = cell_id
        self.content = []


