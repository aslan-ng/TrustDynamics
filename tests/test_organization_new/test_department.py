import unittest
from trustdynamics.organization_new.team import Team


class TestTeam(unittest.TestCase):

    def setUp(self):
        self.team = Team(name="TestTeam")
        self.agent_id_1 = self.team.add_agent(name="TestAgent1")
        self.agent_id_2 = self.team.add_agent(name="TestAgent2")

    def test_all_agent_ids(self):
        all_agent_ids = self.team.all_agents_in_department_ids
        expected = {self.agent_id_1, self.agent_id_2}
        self.assertSetEqual(all_agent_ids, expected)

    def test_all_agent_names(self):
        all_agent_names = self.team.all_agents_in_department_names
        expected = {"TestAgent1", "TestAgent2"}
        self.assertSetEqual(all_agent_names, expected)
    

if __name__ == "__main__":
    unittest.main()
