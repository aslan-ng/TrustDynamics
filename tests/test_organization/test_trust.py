import unittest
from trustdynamics.organization.organization import Organization
from trustdynamics.organization.trust import OrganizationTrust


class TestOrganizationTrust(unittest.TestCase):

    def setUp(self):
        org = Organization(name="TestOrg", ceo_name="Chris")
        self.head_a = org.add_agent(name="Alice", parent=0, department="R&D")
        self.head_b = org.add_agent(name="Bob", parent=0, department="Sales")
        self.comm = OrganizationTrust(org, seed=42)

    def test_departmental_adjacency_dataframe(self):
        df = self.comm.departmental_adjacency_dataframe()
        expected_order = sorted(self.comm.G_departmental.nodes())

        self.assertEqual(df.index.tolist(), expected_order)
        self.assertEqual(df.columns.tolist(), expected_order)

        for src, dst, data in self.comm.G_departmental.edges(data=True):
            self.assertAlmostEqual(df.loc[src, dst], data.get("trust", 0.0))

        # No direct edge between department heads, should fall back to none_value (0.0)
        self.assertEqual(df.loc[self.head_a, self.head_b], 0.0)


if __name__ == "__main__":
    unittest.main()
