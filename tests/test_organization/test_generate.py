import unittest
from trustdynamics.organization.generate import generate_organization


class TestOrganizationGeneration(unittest.TestCase):

    def test_org_size(self):
        n_departments = 3
        n_people = 20
        max_depth = 5
        itter = 10
        found = False
        for _ in range(itter):
            org = generate_organization(n_departments, n_people, max_depth)
            self.assertEqual(org.population, n_people)
            self.assertEqual(len(org.departments()), n_departments)
            if org.depth == max_depth:
                found = True
        self.assertTrue(found)


if __name__ == "__main__":
    unittest.main()
