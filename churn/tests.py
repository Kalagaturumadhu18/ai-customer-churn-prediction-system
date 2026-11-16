from django.test import SimpleTestCase
from churn.ml_utils import required_columns

class ChurnSchemaTests(SimpleTestCase):
    def test_required_columns_are_defined(self):
        self.assertIn("tenure", required_columns())
        self.assertIn("MonthlyCharges", required_columns())
        self.assertNotIn("Churn", required_columns())
