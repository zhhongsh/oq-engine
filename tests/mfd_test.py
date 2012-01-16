import unittest

from nhe.mfd import EvenlyDiscretized, MFDError, TruncatedGR
from nhe.mfd.base import BaseMFD


class BaseMFDTestCase(unittest.TestCase):
    class BaseTestMFD(BaseMFD):
        PARAMETERS = ()
        MODIFICATIONS = set()
        check_constraints_call_count = 0
        def check_constraints(self):
            self.check_constraints_call_count += 1
        def get_annual_occurrence_rates(self):
            pass

    def assert_mfd_error(self, mfd_class, *args, **kwargs):
        with self.assertRaises(MFDError) as exc_catcher:
            mfd_class(*args, **kwargs)
        return exc_catcher.exception


class BaseMFDSetParametersTestCase(BaseMFDTestCase):
    def test_missing(self):
        class TestMFD(self.BaseTestMFD):
            PARAMETERS = ('foo', 'bar', 'baz')
        exc = self.assert_mfd_error(TestMFD, foo=1)
        self.assertEqual(exc.message,
                         'These parameters are required but missing: bar, baz')

    def test_unexpected(self):
        class TestMFD(self.BaseTestMFD):
            PARAMETERS = ('foo', 'bar')
        exc = self.assert_mfd_error(TestMFD, foo=1, bar=2, baz=3)
        self.assertEqual(exc.message, 'These parameters are unexpected: baz')

    def test_check_constraints_is_called(self):
        class TestMFD(self.BaseTestMFD):
            PARAMETERS = ('foo', 'bar')
        mfd = TestMFD(foo=1, bar=2)
        self.assertEqual(mfd.check_constraints_call_count, 1)

    def test_parameters_are_assigned(self):
        class TestMFD(self.BaseTestMFD):
            PARAMETERS = ('baz', 'quux')
        mfd = TestMFD(baz=1, quux=True)
        self.assertEqual(mfd.baz, 1)
        self.assertEqual(mfd.quux, True)


class BaseMFDModificationsTestCase(BaseMFDTestCase):
    def test_modify_missing_method(self):
        class TestMFD(self.BaseTestMFD):
            MODIFICATIONS = ('foo', 'bar')
        mfd = TestMFD()
        exc = self.assert_mfd_error(mfd.modify, 'baz', {})
        self.assertEqual(exc.message,
                         'Modification baz is not supported by TestMFD')

    def test_modify(self):
        class TestMFD(self.BaseTestMFD):
            MODIFICATIONS = ('foo', )
            foo_calls = []
            def modify_foo(self, **kwargs):
                self.foo_calls.append(kwargs)
        mfd = TestMFD()
        self.assertEqual(mfd.check_constraints_call_count, 1)
        mfd.modify('foo', dict(a=1, b='2', c=True))
        self.assertEqual(mfd.foo_calls, [{'a': 1, 'b': '2', 'c': True}])
        self.assertEqual(mfd.check_constraints_call_count, 2)

    def test_reset(self):
        class TestMFD(self.BaseTestMFD):
            PARAMETERS = ('abc', 'defg')
        mfd = TestMFD(abc=1, defg=None)
        mfd.abc = 3
        mfd.defg = []
        mfd.reset()
        self.assertEqual(mfd.abc, 1)
        self.assertEqual(mfd.defg, None)
        self.assertEqual(mfd.check_constraints_call_count, 2)


class EvenlyDiscretizedMFDConstraintsTestCase(BaseMFDTestCase):
    def test_empty_occurrence_rates(self):
        self.assert_mfd_error(
            EvenlyDiscretized,
            min_mag=1, bin_width=2, occurrence_rates=[]
        )

    def test_negative_occurrence_rate(self):
        self.assert_mfd_error(
            EvenlyDiscretized,
            min_mag=1, bin_width=2, occurrence_rates=[-0.1, 1]
        )

    def test_negative_minimum_magnitude(self):
        self.assert_mfd_error(
            EvenlyDiscretized,
            min_mag=-1, bin_width=2, occurrence_rates=[0.1, 1]
        )

    def test_negative_bin_width(self):
        self.assert_mfd_error(
            EvenlyDiscretized,
            min_mag=1, bin_width=-2, occurrence_rates=[0.1, 1]
        )


class EvenlyDiscretizedMFDGetRatesTestCase(BaseMFDTestCase):
    def test_zero_min_width(self):
        mfd = EvenlyDiscretized(min_mag=0, bin_width=1, occurrence_rates=[1])
        self.assertEqual(mfd.get_annual_occurrence_rates(), [(0, 1)])

    def test(self):
        evenly_discretized = EvenlyDiscretized(
            min_mag=0.2, bin_width=0.3, occurrence_rates=[2.1, 2.4, 5.3]
        )
        self.assertEqual(evenly_discretized.get_annual_occurrence_rates(),
                         [(0.2, 2.1), (0.5, 2.4), (0.8, 5.3)])


class TruncatedGRConstraintsTestCase(BaseMFDTestCase):
    def test_negative_min_mag(self):
        self.assert_mfd_error(
            TruncatedGR,
            min_mag=-1, max_mag=2, bin_width=0.4, a_val=1, b_val=2
        )

    def test_min_mag_higher_than_max_mag(self):
        self.assert_mfd_error(
            TruncatedGR,
            min_mag=2.4, max_mag=2, bin_width=0.4, a_val=1, b_val=2
        )

    def test_negative_bin_width(self):
        self.assert_mfd_error(
            TruncatedGR,
            min_mag=1, max_mag=2, bin_width=-0.4, a_val=1, b_val=2
        )

    def test_non_positive_b_val(self):
        self.assert_mfd_error(
            TruncatedGR,
            min_mag=1, max_mag=2, bin_width=0.4, a_val=1, b_val=-2
        )
        self.assert_mfd_error(
            TruncatedGR,
            min_mag=1, max_mag=2, bin_width=0.4, a_val=1, b_val=0
        )


class TruncatedGRMFDGetRatesTestCase(BaseMFDTestCase):
    def _test(self, expected_rates, rate_tolerance, **kwargs):
        mfd = TruncatedGR(**kwargs)
        actual_rates = mfd.get_annual_occurrence_rates()
        self.assertEqual(len(actual_rates), len(expected_rates))
        for i, (mag, rate) in enumerate(actual_rates):
            expected_mag, expected_rate = expected_rates[i]
            self.assertAlmostEqual(mag, expected_mag, delta=1e-14)
            self.assertAlmostEqual(rate, expected_rate, delta=rate_tolerance)

    def test_1_different_min_mag_and_max_mag(self):
        expected_rates = [
            (5.5, 2.846049894e-5),
            (6.5, 2.846049894e-6),
            (7.5, 2.846049894e-7),
        ]
        self._test(expected_rates=expected_rates, rate_tolerance=1e-14,
                   min_mag=5.0, max_mag=8.0, bin_width=1.0,
                   a_val=0.5, b_val=1.0)

    def test_2_different_min_mag_and_max_mag(self):
        expected_rates = [
            (5.5, 2.846049894e-5),
            (6.5, 2.846049894e-6),
            (7.5, 2.846049894e-7),
        ]
        self._test(expected_rates=expected_rates, rate_tolerance=1e-14,
                   min_mag=5.1, max_mag=7.9, bin_width=1.0,
                   a_val=0.5, b_val=1.0)

    def test_3_equal_min_mag_and_max_mag(self):
        expected_rates = [(6.5, 2.307675159e-7)]
        self._test(expected_rates=expected_rates, rate_tolerance=1e-14,
                   min_mag=6.5, max_mag=6.5, bin_width=0.1,
                   a_val=0.5, b_val=1.0)

    def test_4_equal_min_mag_and_max_mag(self):
        expected_rates = [(6.5, 2.307675159e-7)]
        self._test(expected_rates=expected_rates, rate_tolerance=1e-14,
                   min_mag=6.49, max_mag=6.51, bin_width=0.1,
                   a_val=0.5, b_val=1.0)


class TruncatedGRMFDRoundingTestCase(BaseMFDTestCase):
    def test(self):
        mfd = TruncatedGR(min_mag=0.61, max_mag=0.94, bin_width=0.1,
                          a_val=1, b_val=2)
        # mag values should be rounded to 0.6 and 0.9 and there
        # should be three bins with the first having center at 0.65
        min_mag, num_bins = mfd._get_min_mag_and_num_bins()
        self.assertAlmostEqual(min_mag, 0.65)
        self.assertEqual(num_bins, 3)
