from typing import List, Union, Dict
from torch._export.serde.type_utils import check, Succeed, Fail, NotSure
from torch._export.serde.schema import _Union, SymInt, SymExpr
from torch.testing._internal.common_utils import (
    run_tests,
    TestCase,
)


class JustATest:
    def __repr__(self) -> str:
        return "JustATest()"


class JustADifferentTest:
    def __repr__(self) -> str:
        return "JustADifferentTest()"


class TestTypeCheckUtils(TestCase):
    """
    Test basic type utils
    """

    def assert_type_check(self, value: object, type: object, expected_type_check_result: object) -> None:
        self.assertEqual(
            check(value, type),
            expected_type_check_result
        )

    def test_type_utils_check_lists(self):
        """
        Tests various list values
        Demonstrates the structure of error messages
        """

        self.assert_type_check([[42, 120], [10, 5]], List[List[int]], Succeed())

        self.assert_type_check(
            ["yada", "hurray"],
            List[int],
            Fail(
                [
                    "['yada', 'hurray'] is not a List[int].",
                    "Expected int from yada, but got str."
                ]
            )
        )

        self.assert_type_check(
            [[42, 120], [10, "im_a_str"]],
            List[List[int]],
            Fail(
                [
                    "[[42, 120], [10, 'im_a_str']] is not a List[List[int]].",
                    "[10, 'im_a_str'] is not a List[int].",
                    "Expected int from im_a_str, but got str."
                ]
            )
        )


    def test_type_utils_check(self) -> None:
        """
        Tests dictionaries, unions, and classes
        """

        self.assert_type_check(
            {"python": "py", "racket": "rkt"}, Union[List[int], Dict[str, str], float],
            Succeed()
        )

        self.assert_type_check(
            {"five": 5},
            Union[List[int], Dict[str, str], float],
            Fail(
                ["{'five': 5} is not of type Union[List[int], Dict[str, str], float]."]
            )
        )

        self.assert_type_check(
            JustATest(),
            JustATest,
            Succeed()
        )

        self.assert_type_check(
            JustATest(),
            JustADifferentTest,
            Fail(["Expected JustADifferentTest from JustATest(), but got JustATest."])
        )

        self.assert_type_check(
            "fourty_two", 42, NotSure(v="fourty_two", t=42)
        )

class TestUnionTypeCheck(TestCase):
    """
    Test type check in _Union
    """

    def test_union(self) -> None:
        with self.assertRaises(AssertionError) as err:
            SymInt.create(as_int="dog")
        self.assertEqual(
            "SymInt expects as_int of type int but received `dog` of type str.",
            str(err.exception)
        )

        sym_expr = SymExpr(expr_str="yada", hint=[])
        # nothing happens
        sym_int = SymInt.create(as_expr=sym_expr)

        with self.assertRaises(AssertionError) as err:
            SymInt.create(as_int=sym_int)
        self.assertEqual(
            "SymInt expects as_int of type int but received `SymInt(as_expr=SymExpr(expr_str='yada', hint=[]), as_int=None)` of type SymInt.",
            str(err.exception)
        )

        with self.assertRaises(AssertionError) as err:
            SymInt.create(as_expr=42)
        self.assertEqual(
            "SymInt expects as_expr of type SymExpr but received `42` of type int.",
            str(err.exception)
        )

if __name__ == '__main__':
    run_tests()
