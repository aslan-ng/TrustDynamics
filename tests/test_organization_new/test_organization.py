import unittest
from trustdynamics.organization.organization import Organization


class TestOrganization(unittest.TestCase):

    def setUp(self):
        self.org = Organization(name="TestOrg", ceo_name="Chris")

    def test_get_agent_id(self):
        agent_id = self.org.get_agent_id("Chris")
        self.assertEqual(agent_id, 0)

    def test_add_agent(self):
        agent_id = self.org.add_agent(name="Alice", parent="Chris", department="R&D")
        self.assertEqual(agent_id, 1)
        
        self.org.add_agent(name="Bob", parent=agent_id, department="R&D")
        self.assertEqual(self.org.get_agent_id("Bob"), 2)
        #self.org.draw()
        
    def test_add_agent_no_name(self):
        agent_id = self.org.add_agent(parent=0, department="department_1")
        self.assertEqual(agent_id, 1)
        
        new_id = self.org.add_agent(parent=agent_id, department="department_1")
        self.assertEqual(new_id, 2)
        #self.org.draw()

    def test_serialization(self):
        agent_id = self.org.add_agent(name="Alice", parent="Chris", department="R&D")
        self.org.add_agent(name="Bob", parent=agent_id, department="R&D")
        serialized = self.org.serialize()
        org = Organization()
        org.deserialize(serialized)
        self.assertTrue(self.org == org)

    def test_depth_population_departments(self):
        # Only CEO
        self.assertEqual(self.org.depth, 0)
        # CEO -> A -> B
        a_id = self.org.add_agent(name="Alice", parent="Chris", department="R&D")
        b_id = self.org.add_agent(name="Bob", parent=a_id, department="R&D")

        self.assertEqual(b_id, 2)
        self.assertEqual(self.org.depth, 2)
        self.assertEqual(self.org.population, 3)
        self.assertEqual(len(self.org.departments()), 1)
        self.assertIn("R&D", self.org.departments())
        self.assertEqual(len(self.org.agents()), 3)
        self.assertEqual(len(self.org.agents("R&D")), 2)
    
    def test_distance(self):
        # Only CEO
        self.assertEqual(self.org.distance(0), 0)

        # CEO -> A -> B
        a_id = self.org.add_agent(name="Alice", parent="Chris", department="R&D")
        b_id = self.org.add_agent(name="Bob", parent=a_id, department="R&D")

        self.assertEqual(self.org.distance(a_id), 1)          # to CEO
        self.assertEqual(self.org.distance(b_id), 2)          # to CEO
        self.assertEqual(self.org.distance(b_id, a_id), 1)    # between peers

    def test_get_agent_department(self):
        a_id = self.org.add_agent(name="Alice", parent="Chris", department="R&D")
        b_id = self.org.add_agent(name="Bob", parent=a_id, department="R&D")
        c_id = self.org.add_agent(name="Charlie", parent=0, department="Sales")

        self.assertEqual(self.org.get_agent_department(a_id), "R&D")
        self.assertEqual(self.org.get_agent_department(b_id), "R&D")
        self.assertEqual(self.org.get_agent_department(c_id), "Sales")
        self.assertEqual(self.org.get_agent_department(999), None)  # Non-existent agent


if __name__ == "__main__":
    unittest.main()
