"""Offline wrapper around the online data-check test definitions.

This file deliberately reuses the assertions from src.data_check.test_data.
The only difference between online and offline execution is the fixture source:
offline tests load local CSV fixtures from tests_offline/data_check/conftest.py.
"""

from src.data_check import test_data as online_test_data


test_column_names = online_test_data.test_column_names
test_neighborhood_names = online_test_data.test_neighborhood_names
test_proper_boundaries = online_test_data.test_proper_boundaries
test_similar_neigh_distrib = online_test_data.test_similar_neigh_distrib
test_row_count = online_test_data.test_row_count
test_price_range = online_test_data.test_price_range
