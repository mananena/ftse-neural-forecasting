import unittest
import numpy as np
from ftse import make_sequences, ar_forecast, metrics


class ForecastingTests(unittest.TestCase):
    def test_window_keeps_first_and_last_target(self):
        x, y = make_sequences(np.arange(10), 6, 10, window=3)
        np.testing.assert_array_equal(x[0, :, 0], [3, 4, 5])
        np.testing.assert_array_equal(y[:, 0], [6, 7, 8, 9])

    def test_window_does_not_see_target_or_future(self):
        a = np.arange(12, dtype=float)
        b = a.copy(); b[8:] += 10000
        xa, _ = make_sequences(a, 8, 12, 3)
        xb, _ = make_sequences(b, 8, 12, 3)
        np.testing.assert_array_equal(xa[0], xb[0])

    def test_ar_has_no_future_leakage(self):
        rng = np.random.default_rng(3)
        a = 100 + np.cumsum(rng.normal(size=100))
        b = a.copy(); b[85:] += 1000
        for diff in (False, True):
            for expanding in (False, True):
                pa = ar_forecast(a, 80, 5, diff, expanding)
                pb = ar_forecast(b, 80, 5, diff, expanding)
                np.testing.assert_allclose(pa[:6], pb[:6])

    def test_differencing_reconstructs_price(self):
        prices = np.arange(100, dtype=float) + 7000
        forecast = ar_forecast(prices, 80, 1, differenced=True)
        np.testing.assert_allclose(forecast, prices[80:], atol=1e-8)

    def test_metrics_in_price_units(self):
        m = metrics(np.array([100., 200.]), np.array([103., 196.]))
        self.assertEqual(m['MAE'], 3.5)
        self.assertEqual(m['MSE'], 12.5)
        with self.assertRaises(ValueError):
            metrics(np.array([1., 2.]), np.array([[1.], [2.]]))


if __name__ == '__main__':
    unittest.main()
